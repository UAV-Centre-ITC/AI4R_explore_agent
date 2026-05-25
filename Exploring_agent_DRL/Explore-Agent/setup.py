from setuptools import setup

setup(name='explore_agent',
      version='1.0',
      packages=['explore_agent', 'explore_agent.envs'],
      install_requires=['numpy==1.24.0',
                        'ray[rllib]==2.5.1',
                        'ray[default]==2.5.1',
                        'gymnasium',
                        'pygame',
                        'pillow',
                        'pydantic==1.10.15'
                        ]
      )
