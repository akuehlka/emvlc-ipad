Compiling information:

Do not use any modification to DYLD_LIBRARY_PATH: it causes a lot of mess.

Instead, create symlinks to the required libraries in the current directory, like this:
lrwxr-xr-x   1 akuehlka  staff    53B Jun 28 08:58 libboost_filesystem.dylib -> /usr/local/boost-1.63.0/lib/libboost_filesystem.dylib
lrwxr-xr-x   1 akuehlka  staff    48B Jun 28 08:58 libboost_regex.dylib -> /usr/local/boost-1.63.0/lib/libboost_regex.dylib
lrwxr-xr-x   1 akuehlka  staff    49B Jun 28 08:58 libboost_system.dylib -> /usr/local/boost-1.63.0/lib/libboost_system.dylib
lrwxr-xr-x   1 akuehlka  staff    55B Jun 28 08:59 libmat.dylib -> /Applications/MATLAB_R2017a.app/bin/maci64/libmat.dylib
lrwxr-xr-x   1 akuehlka  staff    54B Jun 28 09:12 libmx.dylib -> /Applications/MATLAB_R2017a.app/bin/maci64/libmx.dylib

The compilation should have no trouble locating libs and includes, but at execution time you can have problems like this:
dyld: Library not loaded: @rpath/libmat.dylib
  Referenced from: /Users/akuehlka/src/iris-spoofing-detection/bsifcpp/./bsifcpp
  Reason: image not found
Abort trap: 6

The easiest fix I found for this, is to modify the linking of the executable to the local library symlinks, using otool and install_name_tool, as described below.

1) Check the linking of bsifcpp executable:
crabacornius:bsifcpp akuehlka$ otool -L bsifcpp
bsifcpp:
	/usr/local/opt/opencv3/lib/libopencv_stitching.3.2.dylib (compatibility version 3.2.0, current version 3.2.0)
	...
	@rpath/libmx.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libmat.dylib (compatibility version 0.0.0, current version 0.0.0)

2) Modify bsifcpp's linking to point at the library in the local directory:
crabacornius:bsifcpp akuehlka$ install_name_tool -change @rpath/libmat.dylib @executable_path/libmat.dylib bsifcpp

3) Check the result with otool again:
crabacornius:bsifcpp akuehlka$ otool -L bsifcpp
bsifcpp:
	/usr/local/opt/opencv3/lib/libopencv_stitching.3.2.dylib (compatibility version 3.2.0, current version 3.2.0)
	...
  @rpath/libmx.dylib (compatibility version 0.0.0, current version 0.0.0)
	@executable_path/libmat.dylib (compatibility version 0.0.0, current version 0.0.0)

4) repeat the operation for the other MATLAB library:
crabacornius:bsifcpp akuehlka$ install_name_tool -change @rpath/libmx.dylib @executable_path/libmx.dylib bsifcpp

5) final check:
crabacornius:bsifcpp akuehlka$ otool -L bsifcpp
bsifcpp:
	/usr/local/opt/opencv3/lib/libopencv_stitching.3.2.dylib (compatibility version 3.2.0, current version 3.2.0)
	...
	@executable_path/libmx.dylib (compatibility version 0.0.0, current version 0.0.0)
	@executable_path/libmat.dylib (compatibility version 0.0.0, current version 0.0.0)

6) bsifcpp should work now!
