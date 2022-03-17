import setuptools
import re
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open("requirements.txt", "r") as f:
    required = f.read().splitlines()
dependency_links = []
del_ls = []
for i_l in range(len(required)):
    l = required[i_l]
    if l.startswith("-e"):
        dependency_links.append(l.split("-e ")[-1])
        del_ls.append(i_l)
        required.append(l.split("=")[-1])

for i_l in del_ls[::-1]:
    del required[i_l]

setuptools.setup(
    name="pcg_skel",
    version=find_version("pcg_skel", "__init__.py"),
    author="Casey Schneider-mizell",
    author_email="caseys@alleninstitute.org",
    description="Skeletonization using the pychunkedgraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
    include_package_data=True,
    dependency_links=dependency_links,
    url="https://github.com/AllenInstitute/pcg_skel",
    packages=["pcg_skel"],
)
