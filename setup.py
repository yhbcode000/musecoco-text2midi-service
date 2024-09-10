from setuptools import setup, find_packages

setup(
    name='musecoco_text2midi_service',  # Name of your project
    version='0.1.0',  # Initial version of your project
    author='Your Name or Team',  # Replace with your name or your team's name
    author_email='your-email@example.com',  # Replace with your email
    description='A deployable service module for generating MIDI files from textual descriptions using MuseCoco.',
    long_description=open('README.md').read(),  # Use README.md for the long description
    long_description_content_type='text/markdown',  # Specify the content type of the long description
    url='https://github.com/your-repo/musecoco-text2midi-service',  # Replace with the URL to your project
    packages=find_packages(),  # Automatically find all packages in your project
    include_package_data=True,  # Include additional files specified in MANIFEST.in
    install_requires=[
        'numpy>=1.21.0',  # For numerical operations
        'mido>=1.2.10',  # For handling MIDI files
        'pydub>=0.25.1',  # For audio processing
        'PyYAML>=6.0',  # For handling YAML configurations
        'flask>=2.2.3',  # Flask for building APIs
        'requests>=2.26.0',  # For making HTTP requests
        # Add other dependencies specific to your MIDI generation logic
    ],
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
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.10',  # Minimum version requirement of Python
    entry_points={
        'console_scripts': [
            'start-midi-service=musecoco_text2midi_service.main:main',  # Entry point for your service
        ],
    },
)
