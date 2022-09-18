# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.distributed.distributed_c10d import (
    get_global_rank,
    get_world_size,
)
from torch.fx.experimental.proxy_tensor import (
    get_proxy_slots,
    make_fx,
    proxy_slot,
)
from torch.testing._internal.common_utils import run_tests
from torch.distributed._spmd.comm_tensor import _get_tracer

from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

import spmd
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
)
from spmd.tensor import (
    _Partial,
    DTensor,
    DeviceMesh,
    Replicate,
    Placement,
    Shard,
)
from spmd.tensor.dispatch import operator_dispatch

from dataclasses import dataclass
import copy
from functools import partial
from typing import List, Sequence


class TraceDeviceMeshTestBase:
    def _test_tracing_all_reduce_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                reduced_tensor = mesh.all_reduce(tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return reduced_tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            reduced_tensor = traced_fn(local_tensor)
            res_num = sum(global_ranks)
            self.assertEqual(reduced_tensor, torch.ones(3, 3) * res_num)

    def _test_broadcast_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                received_tensor = mesh.broadcast(tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return received_tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            received_tensor = traced_fn(local_tensor)
            res_num = global_ranks[0]
            self.assertEqual(received_tensor, torch.ones(3, 3) * res_num)

    def _test_scatter_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            scattered_tensors = [
                torch.ones(3, 3, device=self.device_type) * global_rank
                for global_rank in global_ranks
            ]

            def fn(tensors: List[torch.Tensor]):
                received_tensor = mesh.scatter(tensors, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return received_tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)([t + 1 for t in scattered_tensors])

            received_tensor = traced_fn(scattered_tensors)
            self.assertEqual(received_tensor, torch.ones(3, 3) * self.rank)

    def _test_all_gather_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        # each rank have its own tensor, all_gather gives a list
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                gathered_tensors = mesh.all_gather(tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return [t * 1 for t in gathered_tensors]

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            gathered_tensors = traced_fn(local_tensor)
            self.assertEqual(len(gathered_tensors), dim_group_size)
            for idx, gathered_tensor in enumerate(gathered_tensors):
                self.assertEqual(
                    gathered_tensor, torch.ones(3, 3) * global_ranks[idx]
                )


class TraceDeviceMesh3DTest(DistTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_tracing_all_reduce_nd(self):
        self._test_tracing_all_reduce_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_broadcast_nd(self):
        self._test_broadcast_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_scatter_nd(self):
        self._test_scatter_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_all_gather_nd(self):
        self._test_all_gather_nd(torch.arange(8).reshape(2, 2, 2))


class TraceDeviceMesh2DTest(DistTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_tracing_all_reduce_nd(self):
        self._test_tracing_all_reduce_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_broadcast_nd(self):
        self._test_broadcast_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_scatter_nd(self):
        self._test_scatter_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_all_gather_nd(self):
        self._test_all_gather_nd(torch.arange(4).reshape(2, 2))


@dataclass
class DTensorConfigs:
    device_mesh: DeviceMesh
    placements: Sequence[Placement]

class TraceDistTensorTest(DistTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _test_expand(self, xd, yd, f, mesh, out_spec):
        # trace local graph
        x = xd.redistribute(
            device_mesh=mesh, placements=[Replicate()]
        )._local_tensor.clone()
        y = yd.redistribute(
            device_mesh=mesh, placements=[Replicate()]
        )._local_tensor.clone()
        x.requires_grad = xd.requires_grad
        y.requires_grad = yd.requires_grad
        traced_f = make_fx(f)(x, y)

        if self.rank == 0:
            traced_f.graph.print_tabular()

        # map intermediate tensors in traced graph to DTensor objects
        node_to_obj = {}

        # map place holder to real input DTensor objects
        def name_to_input(name):
            return xd if "x" in name else yd

        replacements = {}

        def remap_arg(arg):
            if isinstance(arg, torch.fx.Node):
                obj = node_to_obj[arg]
                if _get_tracer(obj):
                    # This is a shared arg, already has a tracer from last
                    # tracing. Delete the tracer.
                    del obj.__dict__[proxy_slot]
                return obj
            else:
                return arg

        # walk through the traced local graph and expand node with DTensor's
        # dispatch implementation
        for node in traced_f.graph.nodes:
            if node.op == "placeholder":
                node_to_obj[node] = name_to_input(node.name)
            elif isinstance(node.target, torch._ops.OpOverload):
                args = tree_map(remap_arg, node.args)
                #kwargs = tree_map(lambda n: node_to_obj[n], node.kwargs)
                kwargs = node.kwargs

                out = operator_dispatch(
                    node.target,
                    args,
                    kwargs,
                    DTensor._op_to_rules,
                    DTensor._custom_dispatch_ops,
                )
                node_to_obj[node] = out

                def unwrap_local_args(e):
                    if isinstance(e, DTensor):
                        return e._local_tensor
                    else:
                        return e

                def unwrap_dt_configs(e):
                    if isinstance(e, DTensor):
                        return DTensorConfigs(
                            device_mesh=e.device_mesh,
                            placements=e.placements,
                        )
                    else:
                        return e

                local_args = tree_map(unwrap_local_args, args)
                dt_configs = tree_map(unwrap_dt_configs, args)

                def dispatch_with_local_tensors(
                    local_args,
                    kwargs=None,
                    op_call=None,
                    dt_configs=None,
                ):
                    flatten_local_args, spec = tree_flatten(local_args)
                    flatten_dt_configs, spec = tree_flatten(dt_configs)

                    flatten_dt_args = []
                    for local_arg, dt_config in zip(flatten_local_args, flatten_dt_configs):
                        flatten_dt_args.append(
                            DTensor.from_local(
                                local_arg,
                                device_mesh=dt_config.device_mesh,
                                placements=dt_config.placements,
                            )
                        )
                    dt_args = tree_unflatten(flatten_dt_args, spec)

                    return operator_dispatch(
                        op_call,
                        dt_args,
                        kwargs,
                        DTensor._op_to_rules,
                        DTensor._custom_dispatch_ops,
                    )

                # avoid tracing node.target, _op_to_rules and
                # _custom_dispatch_ops as placeholders
                dispatch = partial(
                    dispatch_with_local_tensors,
                    # HACK
                    kwargs=kwargs,
                    op_call=node.target,
                    dt_configs=dt_configs,
                )
                #dispatch_f = partial(dispatch, node.target)
                # trace DTensor's dispatch logic
                print("==== before dispatch make_fx")
                replacements[node] = make_fx(dispatch)(local_args)

                if self.rank == 0 and "copy_" in node.target.__name__:
                    print("---- copy subgraph is ")
                    replacements[node].graph.print_tabular()

            elif node.op == "output":
                # do nothing, its args will be replaced by dispatcher's
                # output in the next for loop
                pass
            else:
                raise ValueError(f"Unrecognized node {node}")

        # replace enodes in local traced graph with DTensor's dispatch graph
        for node in traced_f.graph.nodes:
            if node not in replacements:
                continue

            traced_dispatch = replacements[node]
            # Map DT's dispatch graph input placeholder nodes to the ones in
            # local traced graph. It uses index-based accessing, which is
            # brittle, just for testing purpose.
            flatten_args, _ = tree_flatten(node.args)
            #if self.rank == 0:
                #print("===== node ", node.op, node.target, node.args, node.kwargs)
                #traced_dispatch.graph.lint()
                #traced_dispatch.graph.eliminate_dead_code()
                #traced_dispatch.graph.print_tabular()
            i = 0
            value_remap = {}
            for dtn in traced_dispatch.graph.nodes:
                if dtn.op == "placeholder":
                    value_remap[dtn] = flatten_args[i]
                    i += 1

            # insert DT's dispatch graph to traced local graph.
            with traced_f.graph.inserting_before(node):
                for dtn in traced_dispatch.graph.nodes:
                    if dtn.op == "placeholder":
                        # do nothing, ignore placeholders.
                        pass
                    elif dtn.op == "output":
                        assert (
                            len(dtn.args) == 1 and len(dtn.args[0]) == 1
                        ), f"Expecting single output, but got {dtn.args}"
                        node.replace_all_uses_with(value_remap[dtn.args[0][0]])
                    else:
                        value_remap[dtn] = traced_f.graph.node_copy(
                            dtn, lambda n: value_remap[n]
                        )

            if self.rank == 0:
                print(f"\n=====after replacing {node.target.__name__}\n")
                traced_f.graph.print_tabular()

        traced_f.graph.lint()
        traced_f.graph.eliminate_dead_code()

        zd = DTensor(
            traced_f(xd._local_tensor, yd._local_tensor), mesh, out_spec
        )
        z = zd.redistribute(
            device_mesh=mesh, placements=[Replicate()]
        ).to_local()

        if self.rank == 0:
            print(z, f(x, y))
            traced_f.graph.print_tabular()
        #self.assertEqual(z, f(x, y))

    @with_comms
    def test_simple_expand_replicate_tensor(self):
        def f(x, y):
            return x + y

        mesh = DeviceMesh(self.device_type, torch.arange(2))
        xd = DTensor.from_local(torch.ones(10, 10), mesh, [Replicate()])
        yd = DTensor.from_local(torch.ones(10, 10), mesh, [Replicate()])
        out_spec = [Replicate()]
        self._test_expand(xd, yd, f, mesh, out_spec)

    @with_comms
    def test_simple_expand_shard_replicate_tensor(self):
        def f(x, y):
            return x.matmul(y)

        mesh = DeviceMesh(self.device_type, torch.arange(2))

        xd = DTensor.from_local(torch.ones(10, 10), mesh, [Shard(0)])
        yd = DTensor.from_local(torch.ones(10, 10), mesh, [Replicate()])
        out_spec = [Shard(0)]
        self._test_expand(xd, yd, f, mesh, out_spec)

    @with_comms
    def test_replicate_backward(self):
        def f(x, y):
            z = x + y
            z.sum().backward()
            with torch.no_grad():
                out = y + y.grad
            return out

        mesh = DeviceMesh(self.device_type, torch.arange(2))

        xd = DTensor.from_local(torch.ones(10, 10), mesh, [Replicate()])
        yd = DTensor.from_local(torch.ones(10, 10, requires_grad=True), mesh, [Replicate()])
        # y += y.grad makes y a partial, as y.grad is partial
        out_spec = [_Partial()]
        self._test_expand(xd, yd, f, mesh, out_spec)

    @with_comms
    def test_simple_expand_replicate_shard_tensor(self):
        def f(x, y):
            z = x.matmul(y)
            z.sum().backward()
            return y + y.grad

        mesh = DeviceMesh(self.device_type, torch.arange(2))

        xd = DTensor.from_local(torch.ones(10, 10), mesh, [Shard(0)])
        yd = DTensor.from_local(torch.ones(10, 10, requires_grad=True), mesh, [Replicate()])
        out_spec = [Shard(0)]
        if self.rank == 1:
            import time
            time.sleep(2)
        self._test_expand(xd, yd, f, mesh, out_spec)


if __name__ == "__main__":
    run_tests()
