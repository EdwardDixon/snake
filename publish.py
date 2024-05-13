import re
import requests
import subprocess
from packaging import version
from pathlib import Path

def get_current_version():
    setup_path = Path('setup.py')
    setup_contents = setup_path.read_text()
    version_match = re.search(r"version=['\"]([^'\"]+)['\"]", setup_contents)
    if version_match:
        return version_match.group(1)
    raise ValueError("Version not found in setup.py.")

def get_published_version(package_name):
    response = requests.get(f'https://pypi.org/pypi/{package_name}/json')
    if response.status_code == 200:
        data = response.json()
        return data['info']['version']
    else:
        return None  # If the package does not exist yet on PyPI

def publish_package():
    subprocess.run(['python', 'setup.py', 'sdist', 'bdist_wheel'], check=True)
    subprocess.run(['twine', 'upload', 'dist/*'], check=True)

def main():
    package_name = 'torch-snake'
    current_version = get_current_version()
    published_version = get_published_version(package_name)

    if published_version is None:
        print(f"{package_name} not found on PyPI. Proceeding with first-time upload.")
        publish_package()
        return

    if version.parse(current_version) > version.parse(published_version):
        print(f"Current version {current_version} is greater than published version {published_version}. Uploading new version.")
        publish_package()
    else:
        print(f"Current version {current_version} is not greater than published version {published_version}. Aborting upload.")

if __name__ == "__main__":
    main()
