# setup.py — make sure this is present and used
import os
from setuptools import setup, find_packages

package_name = "ros2_gsam2"

def package_resource_files(root_dir: str):
    entries = []
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
        if not filenames:
            continue
        files = [os.path.join(dirpath, f) for f in filenames]
        rel = os.path.relpath(dirpath, root_dir)
        dest = os.path.join("share", package_name, root_dir if rel == "." else os.path.join(root_dir, rel))
        entries.append((dest, files))
    return entries

data_files = [
    ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
    (f"share/{package_name}", ["package.xml"]),
    (f"share/{package_name}/launch", ["launch/gsam2_server.launch.py"]),
]
data_files += package_resource_files("resources")

setup(
    name=package_name.replace("_", "-"),
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    entry_points={"console_scripts": ["infer_server = ros2_gsam2.infer_server:main"]},
    install_requires=["setuptools"],
    zip_safe=True,
)
