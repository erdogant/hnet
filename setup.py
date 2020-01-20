import setuptools
import re

# versioning ------------
VERSIONFILE="hnet/__init__.py"
getversion = re.search( r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M)
if getversion:
    new_version = getversion.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Setup ------------
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=['d3graph','matplotlib','numpy','pandas','statsmodels','networkx','community','python-louvain','tqdm','sklearn','ismember','imagesc'],
     python_requires='>=3',
     name='hnet',
     version=new_version,
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="Graphical Hypergeometric Networks",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/erdogant/hnet",
	 download_url = 'https://github.com/erdogant/hnet/archive/'+new_version+'.tar.gz',
     packages=setuptools.find_packages(), # Searches throughout all dirs for files to include
     include_package_data=True, # Must be true to include files depicted in MANIFEST.in
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent",
     ],
 )
