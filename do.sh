rm -rf build devel
catkin_make;
rm -rf /home/nvidia/way/catkin_ws_centerpoint/devel/lib/centerpoint;
cp -r -p /home/nvidia/way/catkin_ws_centerpoint/build/centerpoint  /home/nvidia/way/catkin_ws_centerpoint/devel/lib;
