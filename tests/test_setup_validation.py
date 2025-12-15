"""
Validation tests to ensure the testing infrastructure is set up correctly.
"""
import pytest
import numpy as np
import torch
import os
from pathlib import Path


class TestSetupValidation:
    """Test class to validate the testing setup."""
    
    @pytest.mark.unit
    def test_pytest_is_working(self):
        """Verify that pytest is running correctly."""
        assert True
    
    @pytest.mark.unit
    def test_numpy_import(self):
        """Verify numpy is installed and working."""
        arr = np.array([1, 2, 3])
        assert arr.shape == (3,)
        assert np.sum(arr) == 6
    
    @pytest.mark.unit
    def test_torch_import(self):
        """Verify PyTorch is installed and working."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert tensor.shape == torch.Size([3])
        assert torch.sum(tensor).item() == 6.0
    
    @pytest.mark.unit
    def test_fixtures_available(self, temp_dir, mock_config, sample_state, sample_action):
        """Verify that pytest fixtures are working."""
        # Test temp_dir fixture
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "model_name" in mock_config
        assert mock_config["model_name"] == "test_model"
        
        # Test sample_state fixture
        assert isinstance(sample_state, np.ndarray)
        assert sample_state.shape == (24,)
        assert sample_state.dtype == np.float32
        
        # Test sample_action fixture
        assert isinstance(sample_action, np.ndarray)
        assert sample_action.shape == (2,)
        assert sample_action.dtype == np.float32
    
    @pytest.mark.unit
    def test_device_fixture(self, device):
        """Verify the device fixture works correctly."""
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]
    
    @pytest.mark.unit
    def test_sample_batch_fixture(self, sample_batch):
        """Verify the sample batch fixture provides correct data."""
        assert "states" in sample_batch
        assert "actions" in sample_batch
        assert "rewards" in sample_batch
        assert "next_states" in sample_batch
        assert "dones" in sample_batch
        
        batch_size = sample_batch["states"].shape[0]
        assert batch_size == 32
        assert sample_batch["states"].shape == (32, 24)
        assert sample_batch["actions"].shape == (32, 2)
        assert sample_batch["rewards"].shape == (32, 1)
        assert sample_batch["next_states"].shape == (32, 24)
        assert sample_batch["dones"].shape == (32, 1)
    
    @pytest.mark.unit
    def test_random_seed_reset(self):
        """Verify random seeds are reset between tests."""
        # First random number should be deterministic due to seed reset
        np_random = np.random.randn()
        torch_random = torch.randn(1).item()
        
        # These should be the same values each time due to the autouse fixture
        expected_np = np.random.RandomState(42).randn()
        torch.manual_seed(42)
        expected_torch = torch.randn(1).item()
        
        assert np.isclose(np_random, expected_np)
        assert np.isclose(torch_random, expected_torch, rtol=1e-5)
    
    @pytest.mark.unit
    def test_capture_stdout_fixture(self, capsys):
        """Verify stdout capture works (using pytest's built-in capsys)."""
        print("Test output")
        print("Another line")
        
        captured = capsys.readouterr()
        assert "Test output" in captured.out
        assert "Another line" in captured.out
    
    @pytest.mark.unit
    def test_project_structure(self):
        """Verify the expected project structure exists."""
        workspace = Path("/workspace")
        
        # Check main directories
        assert workspace.exists()
        assert (workspace / "TD3").exists()
        assert (workspace / "tests").exists()
        assert (workspace / "tests" / "unit").exists()
        assert (workspace / "tests" / "integration").exists()
        
        # Check test init files
        assert (workspace / "tests" / "__init__.py").exists()
        assert (workspace / "tests" / "unit" / "__init__.py").exists()
        assert (workspace / "tests" / "integration" / "__init__.py").exists()
        
        # Check configuration files
        assert (workspace / "pyproject.toml").exists()
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker can be used."""
        import time
        time.sleep(0.1)  # Simulate slow test
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker can be used."""
        assert True
    
    @pytest.mark.unit
    def test_mock_ros_fixtures(self, mock_ros_node, mock_gazebo_services, mock_publishers, mock_subscribers):
        """Verify ROS-related mock fixtures are available."""
        assert "init_node" in mock_ros_node
        assert "rate" in mock_ros_node
        assert "rate_instance" in mock_ros_node
        
        assert "reset_proxy" in mock_gazebo_services
        assert "pause_proxy" in mock_gazebo_services
        assert "unpause_proxy" in mock_gazebo_services
        
        assert "cmd_vel" in mock_publishers
        assert "model_state" in mock_publishers
        assert "marker" in mock_publishers
        
        assert "scan" in mock_subscribers
        assert "odom" in mock_subscribers


@pytest.mark.unit
def test_module_level_test():
    """Test that module-level tests are discovered."""
    assert 2 + 2 == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])