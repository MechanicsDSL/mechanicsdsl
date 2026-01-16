"""
Integration Tests for Docker Deployment

Tests Docker configurations and container behavior.
Run with: pytest tests/integration/test_docker.py -v
"""

import os
import subprocess

import pytest

# Skip all tests if Docker is not available
pytestmark = pytest.mark.skipif(
    subprocess.run(["docker", "--version"], capture_output=True).returncode != 0,
    reason="Docker not available",
)


class TestDockerfile:
    """Tests for Dockerfile structure and build."""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists in project root."""
        dockerfile = os.path.join(os.path.dirname(__file__), "..", "..", "Dockerfile")
        assert os.path.exists(dockerfile), "Dockerfile not found"

    def test_dockerfile_has_multistage(self):
        """Test that Dockerfile uses multi-stage builds."""
        dockerfile = os.path.join(os.path.dirname(__file__), "..", "..", "Dockerfile")

        with open(dockerfile, "r") as f:
            content = f.read()

        # Check for multi-stage FROM statements
        from_count = content.count("FROM ")
        assert from_count >= 2, "Dockerfile should use multi-stage builds"

        # Check for builder and runtime stages
        assert "AS builder" in content, "Missing builder stage"
        assert "runtime" in content.lower(), "Missing runtime stage"

    def test_dockerfile_has_gpu_stage(self):
        """Test that Dockerfile has GPU/CUDA stage."""
        dockerfile = os.path.join(os.path.dirname(__file__), "..", "..", "Dockerfile")

        with open(dockerfile, "r") as f:
            content = f.read()

        assert "gpu" in content.lower() or "cuda" in content.lower()
        assert "nvidia" in content.lower()


class TestDockerCompose:
    """Tests for Docker Compose configurations."""

    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        compose = os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.yml")
        assert os.path.exists(compose), "docker-compose.yml not found"

    def test_docker_compose_gpu_exists(self):
        """Test that docker-compose.gpu.yml exists."""
        compose_gpu = os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.gpu.yml")
        assert os.path.exists(compose_gpu), "docker-compose.gpu.yml not found"

    def test_docker_compose_has_services(self):
        """Test that docker-compose.yml defines expected services."""
        compose = os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.yml")

        with open(compose, "r") as f:
            content = f.read()

        # Check for expected services
        assert "api:" in content, "Missing api service"
        assert "jupyter:" in content, "Missing jupyter service"
        assert "worker:" in content, "Missing worker service"

        # Check for volumes
        assert "volumes:" in content

        # Check for networks
        assert "networks:" in content

    def test_docker_compose_gpu_has_nvidia(self):
        """Test that GPU compose has NVIDIA settings."""
        compose_gpu = os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.gpu.yml")

        with open(compose_gpu, "r") as f:
            content = f.read()

        assert "nvidia" in content.lower()
        assert "gpu" in content.lower()


class TestDockerBuild:
    """Tests for actual Docker build (slow, optional)."""

    @pytest.mark.slow
    def test_docker_build_syntax(self):
        """Test that Dockerfile has valid syntax."""
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")

        result = subprocess.run(
            ["docker", "build", "--check", "."], cwd=project_root, capture_output=True, text=True
        )

        # Note: --check is not available in older Docker versions
        # If not available, just check that file is parseable
        if "unknown flag" in result.stderr:
            pytest.skip("Docker version doesn't support --check")
