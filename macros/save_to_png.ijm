inputdir = getDir("");

for (i=0; i<=53; i+=1) {
	
	filepath = inputdir + toString(i)+"/";

	files = getFileList(filepath);
	
	filepath1 = filepath+files[0];
	filepath2 = filepath+files[1];
	
	full_params1 = "filepath=[" + filepath1 + "] fileformat=[CSV (comma separated)] livepreview=false rawimagestack= startingframe=1 append=false";
	full_params2 = "filepath=[" + filepath2 + "] fileformat=[CSV (comma separated)] livepreview=false rawimagestack= startingframe=1 append=false";
	
	run("Import results",full_params1);
	run("Visualization", "imleft=0.0 imtop=0.0 imwidth=512.0 imheight=512.0 renderer=[Averaged shifted histograms] magnification=5.7 colorizez=false threed=false shifts=2");
	selectWindow("Averaged shifted histograms");
	rename("w1");
	
	run("Import results", full_params2);
	run("Visualization", "imleft=0.0 imtop=0.0 imwidth=512.0 imheight=512.0 renderer=[Averaged shifted histograms] magnification=5.7 colorizez=false threed=false shifts=2");
	selectWindow("Averaged shifted histograms");
	rename("w2");
	
	run("Merge Channels...", "c1=w1 c2=w2 create");
	run("Stack to RGB");
	close("Composite");
	
	//run("Brightness/Contrast...");
	run("Enhance Contrast", "saturated=0.35");
	
	savepath = inputdir+"STORM_training_set/"+files[0]+".png";

	saveAs("PNG", savepath);
	close();
	
}