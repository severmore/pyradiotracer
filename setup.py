from setuptools import setup, find_packages

with open('README.md') as rm:
  readme = rm.read()

with open('LICENSE') as li:
  license_ = li.read()

setup(
  name = 'sample',
  version = '0.0.1',
  desciption = 'Ray tracer for radio and visual waves',
  long_description = readme,
  author = 'Ivanov Roman',
  author_email = 'iromcorp@gmail.com',
  url = 'https://github.com/severmore/pyradiotracer',
  license = license_,
  packages = find_packages(exclude=('tests', 'docs'))
)