%% Play movies from neuron data
% movie directory
mvdir = 'C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\NeuralRecordings\data\crcns-ringach-data\movie_frames\'
mv = 'movie000_000.images\'
images_seq_to_video('mv.avi',[mvdir mv],'jpeg');