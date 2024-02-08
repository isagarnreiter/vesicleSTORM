#target_dir =  filedialog.askdirectory()
target_dir = '/Users/isabellegarnreiter/Desktop/PSD'
filename = 'PSDstorm_data' 

#name of the widefield image in each folder. 
widefield_image = ''

#folder path in which to find the demixed storm files.
glob_folder_pattern = '*/CellZone*/*emix'

#what to match the files by to create the dataframe. If you want to use all the files in the directory, leave as an empty string.
match = ''

#update lists for markers and DIVs
markers = ['SPON647', 'DEP647', 'PSD680', 'Bassoon680', 'VAMP2680', 'VGLUT647', 'rimbp680', 'Basson647', 'SPON680', 'DEP680']
DIVs = ['6DIV', '8DIV', '10DIV', '13DIV']


params = fcts.get_default_params()
params['647_channel'] = 'channel2'

# usable_exp = pd.read_csv('/users/isabellegarnreiter/documents/vesicleSTORM/data/STORM_binary_list.csv',encoding='latin', sep=',').to_numpy()
# filename  = usable_exp[:,0]+'_'+usable_exp[:,1]
# files_infos = dict(zip(filename, usable_exp[:,2:]))


date_pattern = re.compile(r'^\d')
        
i = 0

nb_points_647 = []
nb_points_680 = []

for folder in os.listdir(target_dir):
    folder_path = os.path.join(target_dir, folder)
    # Check if folder is a directory and if the name starts with a date
    if os.path.isdir(folder_path) and re.compile(r'^\d').match(folder) and match in folder:
        print(folder)

        Demix_folders = glob(folder_path + glob_folder_pattern)
        for demix in Demix_folders:
            if os.path.isdir(demix):
                #MAKE SURE YOU HAVE THE RIGHT CHANNEL HERE.
                if params['647_channel'] == 'channel1':
                    channel680 = glob(demix + '*/*w2*.csv')[0]

                elif params['647_channel'] == 'channel2':
                    channel680 = glob(demix + '*/*w1*.csv')[0]

            data_in_680 = pd.read_csv(channel680)[['x [nm]', 'y [nm]', 'z [nm]']].to_numpy(dtype=np.float64)
            data_in_680[:,2] +=550

            points_647_file = os.path.join(demix, 'data/', 'points_647.npy')
            clusters_647_file = os.path.join(demix, 'data/', 'clusters_647.npy')
            points_647 = np.load(points_647_file, allow_pickle=True)[()]
            clusters_647 = np.load(clusters_647_file, allow_pickle=True)[()]

            points_680_file = os.path.join(demix, 'data/', 'points_647.npy')
            points_680 = np.load(points_680_file, allow_pickle=True)[()]

            for key, value in points_647.items():
                length = len(value)
                nb_points_647.append(length)

            # Process the second dictionary
            for key, value in points_680.items():
                length = len(value)
                nb_points_680.append(length)