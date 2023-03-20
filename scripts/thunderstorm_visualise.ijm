inputdir = getDir("")

experiment = "210414 SPON647_PSD680 8DIV"
cellzone = "CellZone1"

filepath1 = inputdir+"CoordTable_SAFE360_MULTIPLEXING_demixed_w1_UncertaintyFiltered.csv"
filepath2 = inputdir+"CoordTable_SAFE360_MULTIPLEXING_demixed_w2_UncertaintyFiltered.csv"

full_params1 = "filepath=[" + filepath1 + "] fileformat=[CSV (comma separated)] livepreview=false rawimagestack= startingframe=1 append=false"
full_params2 = "filepath=[" + filepath2 + "] fileformat=[CSV (comma separated)] livepreview=false rawimagestack= startingframe=1 append=false"

run("Import results",full_params1);

run("Visualization", "imleft=0.0 imtop=0.0 imwidth=512.0 imheight=512.0 renderer=[Averaged shifted histograms] magnification=5.7 colorizez=false threed=false shifts=2");
selectWindow("Averaged shifted histograms");

rename("w1");

run("Import results", full_params2);
run("Visualization", "imleft=0.0 imtop=0.0 imwidth=512.0 imheight=512.0 renderer=[Averaged shifted histograms] magnification=5.7 colorizez=false threed=false shifts=2");
selectWindow("Averaged shifted histograms");
rename("w2");
run("Merge Channels...", "c1=w1 c2=w2 create");


