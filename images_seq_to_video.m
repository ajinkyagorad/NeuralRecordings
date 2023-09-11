%This function makes a simple avi movie from all image files in a folder
%make sure the image name are ordered correctly
%Usage:
%      Inputs:          
%      output_name: the name of the output video file, by default it is output.avi 
%       images_dir: the directory whre the images are saved, default:
%       current directory
%       img_format: '.bmp'   ,  '.jpg'  , '.jpeg' , '.tif'  default: jpg
%       frame_rate: video's frame rate, default=30
%           
%           Output:
%  it creates a file named as output.avi from the specified image list in the input folder
%           
%            Usage: Try the simple provided example
%   images_seq_to_video(); %--> change the active directory to the image's folder
%   images_seq_to_video('car.avi') ;%--> the output will be car.avi
%   images_seq_to_video('example/car.avi','example'); %--> the images are
%   in the sequence folder under the current active directory
%   images_seq_to_video('car.avi','example','jpg');
%   images_seq_to_video('sequence.avi','example','jpg',10);
%This function is written by :
%                             Kurnianggoro
%                             University of Ulsan
%Tested using MATLAB R2013b

function [ output_args ] = images_seq_to_video( output_name,images_dir,img_format, frame_rate )
%images_seq_to_video( output_name,images_dir,img_format, frame_rate )
%   Detailed explanation goes here
    
    if(nargin<4)
        frame_rate=30;
    end
    if(nargin<3)
        img_format='jpg';
    end
    if(nargin<2)
        images_dir='./';
    end
    if(nargin<1)
        output_name='output.avi';
    end
       
       
    outputVideo = VideoWriter(output_name);
    imgFiles=filesInPath(images_dir,img_format);
    outputVideo.FrameRate=frame_rate;
    open(outputVideo);
    for i=1:length(imgFiles)
        img=imread(imgFiles{i});
        writeVideo(outputVideo,img);
    end
    close(outputVideo);
return


function cellFiles = filesInPath(pathString, varargin)
  if length(varargin) > 0
    extString = lower(varargin{1});
  else
    extString = '.gz';
  end
  dirList = dir(pathString);
  cellFiles = {};
  jx = 1;
  for ix=1:length(dirList)
    fn = lower(dirList(ix).name);
      idxes= findstr(fn, extString);
      if not(isempty(idxes))
	if idxes(end) == (length(fn) - length(extString) + 1)
	  cellFiles{jx} = [pathString '/' fn];
	  jx = jx+1;
	end
    end
  end
  
  filenameSize = size(cellFiles{1});
  allOK = 1;
  for i = 2:length(cellFiles);
    if any(filenameSize ~= size(cellFiles{i}))
      allOK = 0;
    end
  end
  if (allOK)
    %disp('sorting filenames lexicographically');
    for i = 1:length(cellFiles);
      filenames(i,:) = cellFiles{i};
    end
    [jnk,idxes] = sortrows(filenames);
    for i = 1:length(cellFiles);
      cellFiles{i} = filenames(idxes(i),:);
    end
  else
    disp(['filenames of different lengths.  Unable to sort' ...
          ' lexicographically, Matlab inherent order is by creation date']);
  end
  return