from setuptools import setup, find_packages

# Read the contents of the requirements file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='musecoco_text2midi_service',  # Name of your project
    version='0.1.0',  # Initial version of your project
    author='yhbcode000',  # Replace with your name or your team's name
    author_email='hobart.yang@qq.com',  # Replace with your email
    description='A deployable service module for generating MIDI files from textual descriptions using MuseCoco.',
    long_description=open('README.md').read(),  # Use README.md for the long description
    long_description_content_type='text/markdown',  # Specify the content type of the long description
    url='https://github.com/yhbcode000/musecoco-text2midi-service',  # Replace with the URL to your project
    packages=find_packages(where="src"),  # Automatically find all packages in your project
    package_dir={'': 'src'},  # Tell setuptools where the packages are located
    include_package_data=True,  # Include additional files specified in MANIFEST.in
    install_requires=required,
    extras_require={
        'dev': [
            'pytest>=7.0.0',  # For running tests
            'flake8>=4.0.0',  # For code style checking
            'black>=22.3.0',  # For code formatting
        ],
        'test': [
            'pytest>=7.0.0',
            'coverage>=6.0',  # For measuring code coverage
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache2.0 License',
        'Operating System :: Linux',
    ],
    python_requires='>=3.8',  # Minimum version requirement of Python
    entry_points={
        'console_scripts': [
            'start-musecoco-text2midi-service=musecoco_text2midi_service.main:start',  # Entry point for your service
        ],
    },
)
