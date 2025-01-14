# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from spmd.tensor.api import DTensor
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
    TEST_GPU_NUM,
)
from spmd import distribute_tensor, DeviceMesh
from spmd.tensor.placement_types import Placement, Shard, Replicate, _Partial
from typing import List, cast
import itertools


class DistMatrixOpsTest(DistTensorTestBase):
    @with_comms
    def test_addmm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        tensor_to_shard = torch.randn(12, 8)
        mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
        tensor_to_replicate = torch.randn(8, 4)
        mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)
        input_tensor = torch.randn(4)
        input = distribute_tensor(input_tensor, device_mesh, replica_spec)

        dist_res = torch.addmm(input, mat1, mat2)
        local_res = torch.addmm(
            input_tensor, tensor_to_shard, tensor_to_replicate
        )
        self.assertEqual(
            dist_res.redistribute(device_mesh, replica_spec).to_local(),
            local_res,
        )

    @with_comms
    def test_addmm_auto_redistribute(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = [Shard(0)]
        shard1_spec = [Shard(1)]
        replica_spec = [Replicate()]

        tensor_to_shard1 = torch.randn(12, 8, requires_grad=True)
        mat1 = distribute_tensor(tensor_to_shard1, device_mesh, shard1_spec)
        tensor_to_shard0 = torch.randn(8, 4, requires_grad=True)
        mat2 = distribute_tensor(tensor_to_shard0, device_mesh, shard0_spec)
        input_tensor = torch.randn(4, requires_grad=True)
        input = distribute_tensor(input_tensor, device_mesh, replica_spec)

        local_res = torch.addmm(
            input_tensor, tensor_to_shard1, tensor_to_shard0
        )
        dist_res = torch.addmm(input, mat1, mat2)

        # test if addmm output is a partial
        self.assertIsInstance(dist_res, DTensor)
        self.assertIsInstance(dist_res.placements[0], _Partial)

        # test if result is the same as tensor
        replica_res = dist_res.redistribute(device_mesh, replica_spec)
        dist_local_res = replica_res.to_local()
        self.assertEqual(local_res, dist_local_res)

        # backward checks
        dist_local_res.sum().backward()
        local_res.sum().backward()
        self.assertIsNotNone(mat2.grad)
        mat2_grad = mat2.grad.redistribute(device_mesh, replica_spec)
        self.assertEqual(mat2_grad.to_local(), tensor_to_shard0.grad)

    @with_comms
    def test_mm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = [Shard(0)]
        shard1_spec = [Shard(1)]
        replica_spec = [Replicate()]

        t1 = torch.randn(12, 8, requires_grad=True)
        t2 = torch.randn(8, 16, requires_grad=True)
        local_res = torch.mm(t1, t2)

        def test_placement_comb(
            placements1: List[Placement], placements2: List[Placement]
        ) -> None:
            dt1 = distribute_tensor(t1, device_mesh, placements1)
            dt2 = distribute_tensor(t2, device_mesh, placements2)
            dist_res: DTensor = cast(DTensor, torch.mm(dt1, dt2))
            self.assertEqual(
                dist_res.redistribute(device_mesh, replica_spec).to_local(),
                local_res,
            )
            # backward
            grad_dist_res = torch.ones_like(dist_res)
            dist_res.backward(grad_dist_res)
            self.assertIsNotNone(dt1.grad)

        test_placement_comb(replica_spec, replica_spec)
        test_placement_comb(shard0_spec, replica_spec)
        test_placement_comb(replica_spec, shard1_spec)

        # TODO: support (shard1, shard0) -> [partial]
        with self.assertRaises(Exception):
            test_placement_comb(shard1_spec, shard0_spec)

        # TODO: support (shard0, shard1) -> [shard0, shard1]
        with self.assertRaises(Exception):
            test_placement_comb(shard0_spec, shard1_spec)

        with self.assertRaises(Exception):
            test_placement_comb(replica_spec, shard0_spec)

        with self.assertRaises(Exception):
            test_placement_comb(shard1_spec, replica_spec)

    @with_comms
    def test_t(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        tensor_to_transpose = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_transpose, device_mesh, shard_spec)
        tranposed_mat = mat.t()
        self.assertEqual(tranposed_mat.size(), torch.Size([8, 12]))
        self.assertEqual(tranposed_mat.placements, [Shard(1)])
        tranposed_mat2 = tranposed_mat.t()
        self.assertEqual(tranposed_mat2.size(), torch.Size([12, 8]))
        self.assertEqual(tranposed_mat2.placements, shard_spec)

    # baddbmm introduces nan occasionally on CPU: https://github.com/pytorch/pytorch/issues/80588
    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_sharded_baddbmm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        tensor = torch.rand(4, 4, 8, device=self.device_type)
        batch_1 = torch.rand(4, 4, 8, device=self.device_type)
        batch_2 = torch.rand(4, 8, 8, device=self.device_type)

        def test_placement_comb(
            tensor_placements: List[Placement],
            batch_1_placements: List[Placement],
            batch_2_placements: List[Placement],
            beta: int,
            alpha: int,
        ) -> None:
            tensor_dt = distribute_tensor(
                tensor, device_mesh, tensor_placements
            )
            batch_1_dt = distribute_tensor(
                batch_1, device_mesh, batch_1_placements
            )
            batch_2_dt = distribute_tensor(
                batch_2, device_mesh, batch_2_placements
            )
            new_dt = cast(
                DTensor,
                torch.baddbmm(
                    tensor_dt, batch_1_dt, batch_2_dt, beta=beta, alpha=alpha
                ),
            ).redistribute(device_mesh, [Replicate()])
            assert not torch.isnan(local_result).any()
            assert not torch.isnan(new_dt.to_local()).any()
            self.assertEqual(new_dt.to_local(), local_result)

        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        shard2_spec = Shard(2)
        replica_spec = Replicate()
        shard_specs = [shard0_spec, shard1_spec, shard2_spec, replica_spec]
        shard_specs_comb = list(
            itertools.product(shard_specs, shard_specs, shard_specs)
        )
        passlist = [
            (shard0_spec, shard0_spec, shard0_spec),
            (shard0_spec, shard0_spec, replica_spec),
            (shard1_spec, shard1_spec, replica_spec),
            (shard0_spec, replica_spec, shard0_spec),
            (shard2_spec, replica_spec, shard2_spec),
            (replica_spec, shard0_spec, shard0_spec),
            (replica_spec, shard1_spec, replica_spec),
            (replica_spec, shard2_spec, shard1_spec),
            (replica_spec, replica_spec, shard2_spec),
            (replica_spec, replica_spec, replica_spec),
        ]
        # If beta is 0, input tensor will be ignored
        numeric_params_comb = [
            (0.0, 0.5),  # zero-beta
            (0.8, 0.5),  # non-zero-beta
        ]

        for beta, alpha in numeric_params_comb:
            local_result = torch.baddbmm(
                tensor, batch_1, batch_2, beta=beta, alpha=alpha
            )
            # tests that currently pass
            for spec in passlist:
                test_placement_comb(
                    [spec[0]], [spec[1]], [spec[2]], beta, alpha
                )

            # TODO: support these tests
            # TODO: test with replicate
            shard_specs_comb = [
                spec for spec in shard_specs_comb if spec not in passlist
            ]
            for spec in shard_specs_comb:
                with self.assertRaises(Exception):
                    test_placement_comb(
                        [spec[0]], [spec[1]], [spec[2]], beta, alpha
                    )

    @with_comms
    def test_bmm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input = torch.rand(4, 8, 4, device=self.device_type)
        mat_2 = torch.rand(4, 4, 8, device=self.device_type)
        local_result = torch.bmm(input, mat_2)

        def test_placement_comb(
            placements1: List[Placement],
            placements2: List[Placement],
        ) -> None:
            input_dt = distribute_tensor(input, device_mesh, placements1)
            mat_2_dt = distribute_tensor(mat_2, device_mesh, placements2)
            new_dt = cast(DTensor, torch.bmm(input_dt, mat_2_dt)).redistribute(
                device_mesh, [Replicate()]
            )
            self.assertEqual(new_dt.to_local(), local_result)

        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        shard2_spec = Shard(2)
        replica_spec = Replicate()
        placement_specs = [shard0_spec, shard1_spec, shard2_spec, replica_spec]
        shard_specs_comb = list(
            itertools.product(placement_specs, placement_specs)
        )
        passlist = [
            (shard0_spec, shard0_spec),
            (shard0_spec, replica_spec),
            (shard1_spec, replica_spec),
            (shard2_spec, replica_spec),
            (replica_spec, shard0_spec),
            (replica_spec, shard1_spec),
            (replica_spec, shard2_spec),
            (shard2_spec, shard1_spec),
            (replica_spec, replica_spec),
        ]

        # tests that currently pass
        for spec in passlist:
            test_placement_comb([spec[0]], [spec[1]])

        # TODO: support these tests
        shard_specs_comb = [
            spec for spec in shard_specs_comb if spec not in passlist
        ]
        for spec in shard_specs_comb:
            with self.assertRaises(Exception):
                test_placement_comb([spec[0]], [spec[1]])


if __name__ == "__main__":
    run_tests()
