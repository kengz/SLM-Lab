import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

test_args = [
    '-n 0',
    '--verbose',
    '--capture=sys',
    '--log-level=INFO',
    '--log-cli-level=INFO',
    '--log-file-level=INFO',
    '--no-flaky-report',
    '--timeout=300',
    '--cov-report=html',
    '--cov-report=term',
    '--cov-report=xml',
    '--cov=slm_lab/agent',
    '--cov=slm_lab/env',
    '--cov=slm_lab/experiment',
    '--cov=slm_lab/lib',
    '--cov=slm_lab/spec',
    '--ignore=test/agent/net',
    'test',
]


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', 'Arguments to pass to py.test')]

    def initialize_options(self):
        os.environ['PY_ENV'] = 'test'
        TestCommand.initialize_options(self)
        self.pytest_args = test_args

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='slm_lab',
    version='2.1.2',
    description='Modular Deep Reinforcement Learning framework in PyTorch.',
    long_description='https://github.com/kengz/slm_lab',
    keywords='SLM Lab',
    url='https://github.com/kengz/slm_lab',
    author='kengz,lgraesser',
    author_email='kengzwl@gmail.com',
    license='MIT',
    packages=[],
    zip_safe=False,
    include_package_data=True,
    dependency_links=[],
    extras_require={
        'dev': [],
        'docs': [],
        'testing': []
    },
    classifiers=[],
    test_suite='test',
    cmdclass={'test': PyTest},
)
