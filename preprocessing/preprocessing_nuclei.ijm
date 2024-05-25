dir = getDirectory("Choose a data directory ");
list = getFileList(dir);

save_dir = getDirectory("Choose a saved directory ");

for(i=0;i<lengthOf(list);i++){
	open(dir + list[i]);
	run("8-bit");
	run("Enhance Contrast", "saturated=0.35");
	run("Apply LUT");
	image_num = split(list[i], ".");
	run("Size...", "width=2048 height=2048 depth=1 average interpolation=Bilinear");
	for(x=0;x<4;x++){
		for(y=0;y<4;y++){
			makeRectangle(x*512, y*512, 512, 512);
			run("Duplicate...", " ");
			saveAs("Tiff", save_dir + image_num[0] + x + "-" + y + ".tif");
			close();
		}
	}
	close();
}
