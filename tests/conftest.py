"""
Shared pytest fixtures and configuration for all tests.
"""
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary."""
    return {
        "model_name": "test_model",
        "batch_size": 32,
        "learning_rate": 0.001,
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        "max_episodes": 100,
        "max_steps": 1000,
        "start_steps": 1000,
        "save_freq": 5000,
        "seed": 42,
    }


@pytest.fixture
def sample_state():
    """Provide a sample state for testing."""
    return np.random.randn(24).astype(np.float32)


@pytest.fixture
def sample_action():
    """Provide a sample action for testing."""
    return np.array([0.5, 0.3], dtype=np.float32)


@pytest.fixture
def sample_batch():
    """Provide a sample batch of experiences."""
    batch_size = 32
    state_dim = 24
    action_dim = 2
    
    return {
        "states": torch.randn(batch_size, state_dim),
        "actions": torch.randn(batch_size, action_dim),
        "rewards": torch.randn(batch_size, 1),
        "next_states": torch.randn(batch_size, state_dim),
        "dones": torch.randint(0, 2, (batch_size, 1), dtype=torch.float32),
    }


@pytest.fixture
def mock_ros_node():
    """Mock ROS node for testing ROS-dependent code."""
    try:
        with patch("rospy.init_node") as mock_init:
            with patch("rospy.Rate") as mock_rate:
                mock_rate_instance = Mock()
                mock_rate.return_value = mock_rate_instance
                yield {
                    "init_node": mock_init,
                    "rate": mock_rate,
                    "rate_instance": mock_rate_instance,
                }
    except ModuleNotFoundError:
        # ROS not installed, provide mocks directly
        yield {
            "init_node": Mock(),
            "rate": Mock(),
            "rate_instance": Mock(),
        }


@pytest.fixture
def mock_gazebo_services():
    """Mock Gazebo services for testing."""
    services = {
        "reset_proxy": Mock(),
        "pause_proxy": Mock(),
        "unpause_proxy": Mock(),
    }
    
    try:
        with patch("rospy.ServiceProxy") as mock_service:
            def service_side_effect(service_name, service_type):
                if "reset" in service_name:
                    return services["reset_proxy"]
                elif "pause" in service_name:
                    return services["pause_proxy"]
                elif "unpause" in service_name:
                    return services["unpause_proxy"]
                return Mock()
            
            mock_service.side_effect = service_side_effect
            yield services
    except ModuleNotFoundError:
        # ROS not installed, provide mocks directly
        yield services


@pytest.fixture
def mock_publishers():
    """Mock ROS publishers for testing."""
    publishers = {
        "cmd_vel": Mock(),
        "model_state": Mock(),
        "marker": Mock(),
    }
    
    try:
        with patch("rospy.Publisher") as mock_pub:
            def pub_side_effect(topic, msg_type, **kwargs):
                if "cmd_vel" in topic:
                    return publishers["cmd_vel"]
                elif "model_state" in topic:
                    return publishers["model_state"]
                elif "marker" in topic:
                    return publishers["marker"]
                return Mock()
            
            mock_pub.side_effect = pub_side_effect
            yield publishers
    except ModuleNotFoundError:
        # ROS not installed, provide mocks directly
        yield publishers


@pytest.fixture
def mock_subscribers():
    """Mock ROS subscribers for testing."""
    subscribers = {
        "scan": Mock(),
        "odom": Mock(),
    }
    
    try:
        with patch("rospy.Subscriber") as mock_sub:
            def sub_side_effect(topic, msg_type, callback, **kwargs):
                if "scan" in topic:
                    subscribers["scan"].callback = callback
                    return subscribers["scan"]
                elif "odom" in topic:
                    subscribers["odom"].callback = callback
                    return subscribers["odom"]
                return Mock()
            
            mock_sub.side_effect = sub_side_effect
            yield subscribers
    except ModuleNotFoundError:
        # ROS not installed, provide mocks directly
        yield subscribers


@pytest.fixture
def device():
    """Provide the appropriate torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def capture_stdout():
    """Capture stdout for testing print statements."""
    import io
    import sys
    from contextlib import redirect_stdout
    
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        yield captured_output


# Markers for test organization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_ros: Tests that require ROS environment")
    config.addinivalue_line("markers", "requires_gazebo: Tests that require Gazebo simulator")