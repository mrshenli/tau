import torch
import torch.fx as fx
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import get_global_rank, get_world_size
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.testing._internal.common_utils import run_tests
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.utils._pytree import tree_flatten, tree_map
from functorch.compile import aot_module

from dataclasses import dataclass

from spmd.tensor import (
    DTensor,
    DeviceMesh,
    Placement,
    Replicate,
    Shard,
)
from spmd.tensor.dispatch import operator_dispatch, propagate_input_sharding
from spmd.tensor.redistribute import _redistribute_with_local_tensor

import os
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple


@dataclass
class Schema:
    mesh: DeviceMesh
    placements: List[Placement]


def _dispatch_with_local_tensors(
    op: torch._ops.OpOverload,
    local_args: Tuple[object, ...],
    kwargs: Dict[str, object] = {},
    specs: Dict[
        torch.Tensor,
        Tuple[torch.Size, DeviceMesh, Sequence[Placement], Sequence[Placement]],
    ] = {},
) -> Any:
    def redistribute(arg):
        return (
            _redistribute_with_local_tensor(arg, *specs[arg])
            if arg in specs
            else arg
        )

    return op(*tree_map(redistribute, local_args), **kwargs)


class SPMD(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        schema: Schema,
        schema_override: Callable[[str, nn.Module], nn.Module],
    ):
        super().__init__()
        assert schema.placements == [Replicate()], (
            "SPMD only support Replicate() parameters for now"
        )
        self._local_module = module
        self._schema = schema

        # TODO: support schema_override

        self._compiled_m = None

    def _compile_fwd(self, gm: fx.GraphModule, inps: List[torch.Tensor]) -> fx.GraphModule:
        # HACK: use pytree order of params to map to primals, and save the info
        # for compile_bwd.
        def to_param(model: nn.Module, primal_name: str) -> torch.nn.Parameter:
            idx = int(primal_name.split("_")[-1]) - 1
            params = [p for _, p in list(tree_flatten(model.named_parameters())[0][0])]
            return params[idx] if idx < len(params) else None

        def to_dtensor(schema: Schema, t: torch.Tensor) -> DTensor:
                return DTensor.from_local(t, schema.mesh, schema.placements)

        node_to_obj: Dict[fx.Node, object] = {}
        # map local op node in traced_f to its corresponding subgraph of
        # DTensor ops.
        replacements: Dict[torch.fx.Node, torch.fx.GraphModule] = {}

        def remap_arg(arg):
            if isinstance(arg, torch.fx.Node):
                obj = node_to_obj[arg]
                if _get_tracer(obj):
                    # This is a shared arg, already has a tracer from previous
                    # tracing. Delete the tracer.
                    del obj.__dict__[proxy_slot]
                return obj
            else:
                return arg

        inp_idx = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                p = to_param(self._local_module, node.name)
                if p is not None:
                    node_to_obj[node] = DTensor.from_local(
                        p, self._schema.mesh, self._schema.placements
                    )
                else:
                    node_to_obj[node] = DTensor.from_local(
                        inps[inp_idx], self._schema.mesh, [Shard(0)]
                    )
                    inp_idx += 1
            elif isinstance(node.target, torch._ops.OpOverload):
                args = tree_map(remap_arg, node.args)
                # kwargs in this set of tests are all constants
                kwargs = node.kwargs

                # run dispatch once to get the real DTensor output
                out = operator_dispatch(
                    node.target,
                    args,
                    node.kwargs,  # kwargs in this set of tests are all constants
                    DTensor._op_to_rules,
                    DTensor._custom_dispatch_ops,
                )
                node_to_obj[node] = out

                # get DTensor specs for inputs and outputs
                (
                    target_schema,
                    redistribute,
                    output_sharding,
                ) = propagate_input_sharding(
                    node.target,
                    args,
                    kwargs,
                    DTensor._op_to_rules,
                )

                flatten_args, args_tree_spec = tree_flatten(args)
                flatten_args_schema, _ = tree_flatten(target_schema.args_schema)

                specs: Dict[
                    torch.Tensor,
                    Tuple[
                        torch.Size,
                        DeviceMesh,
                        Sequence[Placement],
                        Sequence[Placement],
                    ],
                ] = {}
                for i, arg in enumerate(flatten_args):
                    if isinstance(arg, DTensor) and redistribute:
                        specs[arg._local_tensor] = (
                            arg.size(),
                            flatten_args_schema[i].mesh,
                            arg.placements,
                            flatten_args_schema[i].placements,
                        )

                dispatch = partial(
                    _dispatch_with_local_tensors,
                    node.target,
                    kwargs=kwargs,
                    specs=specs,
                )

                def unwrap_local(e):
                    return e._local_tensor if isinstance(e, DTensor) else e

                replacements[node] = make_fx(dispatch)(
                    tree_map(unwrap_local, args)
                )
            elif node.op == "output":
                # do nothing, its args will be replaced by dispatcher's
                # output in the next for loop
                pass
            else:
                raise ValueError(f"Unrecognized node {node}")


        # replace nodes in local traced graph with DTensor's dispatch graph
        for node in gm.graph.nodes:
            if node not in replacements:
                continue

            traced_dispatch = replacements[node]
            # Map DT's dispatch graph input placeholder nodes to the ones in
            # local traced graph. It uses index-based accessing, which is
            # brittle, just for testing purpose.
            flatten_args, _ = tree_flatten(node.args)
            i, value_remap = 0, {}
            for dtn in traced_dispatch.graph.nodes:
                if dtn.op == "placeholder":
                    value_remap[dtn] = flatten_args[i]
                    i += 1

            # insert DT's dispatch graph to traced local graph.
            with gm.graph.inserting_before(node):
                for dtn in traced_dispatch.graph.nodes:
                    if dtn.op == "placeholder":
                        # do nothing, ignore placeholders, as it has already
                        # been prepared in value_remap
                        pass
                    elif dtn.op == "output":
                        assert (
                            len(dtn.args) == 1 and len(dtn.args[0]) == 1
                        ), f"Expecting single output, but got {dtn.args}"
                        node.replace_all_uses_with(value_remap[dtn.args[0][0]])
                    else:
                        value_remap[dtn] = gm.graph.node_copy(
                            dtn, lambda n: value_remap[n]
                        )

        gm.graph.lint()
        gm.graph.print_tabular()
        # TODO: update placeholder names in backward graph
        gm.graph.eliminate_dead_code()

        return gm

    def _compile_bwd(self, gm: fx.GraphModule, inps: List[torch.Tensor]) -> fx.GraphModule:
        #gm.graph.print_tabular()
        return gm

    def forward(self, *args, **kwargs):
        if self._compiled_m is None:
            self._compiled_m = aot_module(self._local_module, self._compile_fwd, self._compile_bwd)

        return self._compiled_m(*args, **kwargs)



def run(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    m = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
    device = torch.device("cpu")
    schema = Schema(
        mesh=DeviceMesh(device.type, torch.arange(2)),
        placements=[Replicate()],
    )
    spmd = SPMD(m, schema=schema, schema_override=lambda x: x)
    spmd(torch.zeros(2, 10)).sum().backward()


if __name__=="__main__":
    world_size = 2
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )