from setuptools import setup, find_packages

setup(
    name='v8_risk',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pandas'
    ],
    url='https://github.com/v8capital/risk.git',
    author='Rafael Oliveira',
    author_email='rafael.oliveira@v8capital.com.br',
    description='Módulo para cálculo de risco de portfólio, com arquitetura modular e suporte a múltiplas subclasses de ativos.',
)
