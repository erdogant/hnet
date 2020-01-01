import setuptools
import versioneer
new_version='0.1.4'

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=['d3graph','matplotlib','numpy','pandas','statsmodels','networkx','seaborn','community','python-louvain','tqdm','sklearn'],
     python_requires='>=3',
     name='hnet',
     version=new_version,
#     version=versioneer.get_version(),    # VERSION CONTROL
#     cmdclass=versioneer.get_cmdclass(),  # VERSION CONTROL
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
