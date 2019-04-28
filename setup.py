from setuptools import setup
from setuptools import find_packages

setup(name='VisualNN',
      version='1.0',
      description='Deep Learning visualization library',
      author='Vedang Waradpande, Manan Pachchigar',
      author_email='vedang.waradpande@gmail.com',
      download_url='https://github.com/VedangW/VisualNN',
      license='MIT',
      install_requires=['numpy',
                        'keras',
                        'plotly'
                       ],
      packages=find_packages())
