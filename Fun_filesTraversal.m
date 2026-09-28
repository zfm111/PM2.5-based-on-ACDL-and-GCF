function [ myFiles ] = Fun_filesTraversal(strPath,varargin )
%% 变量说明
% strPath：文件位置
% 如果输入了格式限制字符串，则用限制字符串限制输出

%% 格式限制字符串
if isempty(varargin)
    strLim='*';
elseif length(varargin)==1
    strLim=varargin{1};
end

%% 如果路径最后没有斜杠，则增加
if ~strcmp(strPath(end),'\')||~strcmp(strPath(end),'/')
    strPath(end+1)='/';
end

%% 获取后缀
strInd=strfind(strLim,'.');
fileType=strLim(strInd+1:end);

%% 定义两数组，分别保存文件和路径
myFiles = cell(3,0);
myPath  = cell(0,0);

myPath{1}=strPath;
[r,c] = size(myPath);
while c ~= 0
    strPath = myPath{1};

    Files = dir(fullfile(strPath,'*.*')); %初次获取文件，并根据文件判断
    LengthFiles = length(Files);
    if LengthFiles == 0
        break;
    elseif size(Files,1)>2&&Files(3).isdir==0
        Files = dir(fullfile(strPath,strLim)); %重新获取对应格式的文件
        LengthFiles = length(Files);%重新获取对应格式的文件的长度
    end
    
    myPath(1)=[];
    iCount = 1;
    while LengthFiles>0
        if Files(iCount).isdir==1
            if Files(iCount).name ~='.'
                filePath = [strPath  Files(iCount).name '\'];
                [r,c] = size(myPath);
                myPath{c+1}= filePath;
            end
        elseif Files(iCount).isdir==0&&...
                strcmp(Files(iCount).name(end-length(fileType)+1:end),fileType)
            filePath = [strPath  Files(iCount).name];
            [row,col] = size(myFiles);
            myFiles{1,col+1}=filePath;
            myFiles{2,col+1}=strPath;
            myFiles{3,col+1}=Files(iCount).name;
        end
        
        LengthFiles = LengthFiles-1;
        iCount = iCount+1;
    end
    [r,c] = size(myPath);
end

myFiles = myFiles';

end