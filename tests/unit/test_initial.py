import os
import tomllib

import pytest


def test_always_passes():
    """
    A placeholder test to ensure pytest finds and executes at least one test.
    This can be removed or replaced as real tests are added to the project.
    """
    assert True, "This test should always pass."


def test_project_has_readme():
    """
    Checks for the existence of a README.md file in the project's root directory.
    """
    assert os.path.isfile("README.md"), "The project must contain a README.md file."


def test_pyproject_toml_is_valid_and_contains_project_name():
    """
    Verifies that pyproject.toml exists, is a valid TOML file,
    and contains the correct project name.
    """
    pyproject_path = "pyproject.toml"
    assert os.path.isfile(pyproject_path), f"{pyproject_path} not found in the project root."

    try:
        # 'rb' mode is required for tomllib.load()
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Check if the project name matches the expected name
        project_name = data.get("project", {}).get("name")
        expected_name = "crypto-stock-mlops-pipeline"
        assert project_name == expected_name, (
            f"Project name in {pyproject_path} is '{project_name}', but expected '{expected_name}'."
        )

    except tomllib.TOMLDecodeError:
        pytest.fail(f"{pyproject_path} is not a valid TOML file.")
    except Exception as e:
        pytest.fail(f"An error occurred while reading {pyproject_path}: {e}")
