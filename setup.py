import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

test_args = [
    '-n 2',
    '--cov-report=html',
    '--cov-report=term',
    '--cov-report=xml',
    '--cov=unity_lab',
    'test'
]


def read(filepath):
    return open(os.path.join(os.path.dirname(__file__), filepath)).read()


env_file = read('environment.yml')
dep_str = env_file.split('dependencies:')[-1]
conda_dep_str, pip_dep_str = dep_str.split('- pip:')
conda_dep = conda_dep_str.rstrip('\n').split('\n- ')[1:]
conda_as_pip_dep = ['=='.join(c_dep.split('=')[:2]) for c_dep in conda_dep]
pip_dep = pip_dep_str.rstrip('\n').split('\n  - ')[1:]
dependencies = conda_as_pip_dep + pip_dep


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', 'Arguments to pass to py.test')]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = test_args

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='unity_lab',
    version='0.1.0',
    description='An experimental framework for Reinforcement Learning using Unity and PyTorch.',
    long_description=read('README.md'),
    keywords='Unity Lab',
    url='https://github.com/kengz/unity_lab',
    author='kengz,lgraesser',
    author_email='kengzwl@gmail.com',
    license='MIT',
    packages=[],
    zip_safe=False,
    include_package_data=True,
    # install_requires=dependencies,
    dependency_links=[],
    extras_require={
        'dev': [],
        'docs': [],
        'testing': []
    },
    classifiers=[],
    tests_require=['pytest', 'pytest-cov'],
    test_suite='test',
    cmdclass={'test': PyTest},
)
