# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
)
from spmd import distribute_tensor, DeviceMesh, DTensor, Shard, Replicate


class DistTensorOpsTest(DistTensorTestBase):
    @with_comms
    def test_detach(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        tensor_to_detach = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_detach, device_mesh, shard_spec)
        detached_mat = mat.detach()
        self.assertFalse(detached_mat is mat)

    @with_comms
    def test_clone(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        specs = [[Replicate()], [Shard(0)]]
        tensor_to_clone = torch.randn(12, 8, requires_grad=True)
        for spec in specs:
            mat = distribute_tensor(tensor_to_clone, device_mesh, spec)
            cloned_mat = mat.clone()
            self.assertFalse(cloned_mat is mat)
            self.assertEqual(cloned_mat.to_local(), mat.to_local())

    @with_comms
    def test_contiguous(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        tensor = torch.rand(3, 5, 6, requires_grad=True)
        sharding = [Shard(0)]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        self.assertTrue(dist_tensor.is_contiguous())
        # shard on dim 0 should not change stride (30, 6, 1)
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        new_dt = dist_tensor.transpose(0, 2)
        self.assertFalse(new_dt.is_contiguous())
        self.assertFalse(new_dt.to_local().is_contiguous())
        # check stride
        self.assertEqual(new_dt.stride(), (1, 6, 30))

        new_dt = new_dt.contiguous()
        self.assertTrue(new_dt.is_contiguous())
        self.assertTrue(new_dt.to_local().is_contiguous())
        # check stride
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        # check backward
        new_dt.to_local().sum().backward()
        self.assertEqual(tensor.grad, torch.ones(3, 5, 6))

    @with_comms
    def test_inplace_op(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input_tensor = torch.randn((12, 3), device=self.device_type)
        dt_to_add = distribute_tensor(input_tensor, mesh, [Shard(0)])
        dt_to_mul = dt_to_add.clone()
        expected_add_dt = dt_to_add.clone() + 3
        add_res = dt_to_add.add_(3)
        expected_mul_dt = dt_to_mul.clone() * 3
        mul_res = dt_to_mul.mul_(3)
        # inplace op should be the same instance before and after
        self.assertTrue(add_res is dt_to_add)
        self.assertEqual(add_res.to_local(), expected_add_dt.to_local())

        self.assertTrue(mul_res is dt_to_mul)
        self.assertEqual(mul_res.to_local(), expected_mul_dt.to_local())

    @with_comms
    def test_op_out_variant(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input_tensor = torch.randn((12, 3), device=self.device_type)
        dist_tensor_out = distribute_tensor(input_tensor, mesh, [Shard(0)])
        expected_dt = dist_tensor_out.clone() + 3
        res = torch.add(dist_tensor_out, 3, out=dist_tensor_out)
        # op out variant should be the same instance before and after
        self.assertTrue(res is dist_tensor_out)
        self.assertEqual(dist_tensor_out.to_local(), expected_dt.to_local())

    @with_comms
    def test_ones_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        ones_like_dt = torch.ones_like(dist_tensor)
        ones_expected = torch.ones(4, 8)
        self.assertEqual(ones_expected, ones_like_dt.to_local())


if __name__ == "__main__":
    run_tests()
