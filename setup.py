from distutils.core import setup
from setuptools import find_packages

with open('README.md', encoding='utf-8') as file:
    description = file.read()

setup(
    name='gfn',
    version='0.0.1',
    packages=find_packages(),
    license='',
    zip_safe=True,
    description='',
    long_description=description,
    long_description_content_type='text/markdown',
    author='Hieu Pham',
    author_email='64821726+hieupth@users.noreply.github.com',
    url='',
    keywords=[],
    install_requires=['uvicorn[standard]', 'python-multipart'],
    classifiers=[
      'Development Status :: 1 - Planning',
      'Intended Audience :: Developers',
      'Topic :: Software Development :: Build Tools',
      'Programming Language :: Python :: 3'
    ],
)