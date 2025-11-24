from setuptools import setup
import glob
import os

package_name = 'lan_webrtc_hand'

packages = [
    package_name,
    package_name + '/quest',
    package_name + '/server',
    package_name + '/utils',
]

setup(
    name=package_name,
    version='0.0.0',
    packages=packages,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='m',
    maintainer_email='bonnie@limxdynamics.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'quest_sender = lan_webrtc_hand.quest.quest_sender:main',
            'server = lan_webrtc_hand.server.signaling_server:main',
            'hand_teleop = lan_webrtc_hand.quest.hand_teleop:main',
            'hand_offset = lan_webrtc_hand.quest.hand_offset:main',
        ],
    },
)
