//c++ libraries
#include <iostream>
#include <utility>
//opencv libraries
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class Transform_params
{
	public:
	int k,l,direction;
	float angle,contrast,brightness;

	Transform_params() //default contructor
	{
		k = 0;
		l = 0;
		direction = 0;
		angle = 0;
		contrast = 0;
		brightness = 0;
	};

	Transform_params(int i,int j,int dir,float ang,float contr,float bright)
	{
		k = i;
		l = j;
		direction = dir;
		angle = ang;
		contrast = contr;
		brightness = bright;
	};
};

class Transform_block
{
	public:
	Mat img;
	Transform_params params;

	Transform_block()
	{
		Transform_params parameters(0,0,0,0.0,0.0,0.0);
		params = parameters;
	};

	Transform_block(Mat image, Transform_params parameters)
	{
		img = image;
		params = parameters;
	};
};

Mat extract(Mat img, int i, int j, int rows, int cols)
{
	Mat block = Mat::zeros(rows,cols,CV_8UC1);
	for(int k=0;k<rows;k++)
	{
		for(int l=0;l<cols;l++)
		{
			block.at<uchar>(k,l) = img.at<uchar>(i+k,j+l);
		}
	}

	return block;
}

Mat reduce(Mat img, int factor)
{
	int rows = img.rows/factor;
	int cols = img.cols/factor;
	Mat reduced_img = Mat::zeros(rows,cols,CV_8UC1);

	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			for(int k=i*factor;k<(i+1)*factor;k++)
			{
				for(int l=j*factor;l<(j+1)*factor;l++)
				{
					reduced_img.at<uchar>(i,j) += img.at<uchar>(k,l);
				}
			}

			reduced_img.at<uchar>(i,j) = reduced_img.at<uchar>(i,j)/(factor*factor);
		}
	}

	return reduced_img;
}

Mat rotate(Mat img,float angle)
{
	//cout<<"rows is "<<img.rows;
	Point2f centre(img.cols/2.0F,img.rows/2.0F); //create point about which it rotates
	Mat rot_matrix = getRotationMatrix2D(centre, angle, 1.0); //Rotation matrix
	//Mat rotated_img(Size(img.size().height, img.size().width), img.type());
	Mat rotated_img = Mat::zeros(img.rows,img.cols,img.type());
	//cout<<"hi";
	warpAffine(img, rotated_img, rot_matrix, img.size()); //transformation
	//cout<<"hey";
	return rotated_img;
}

pair<float,float> find_contrast_brightness(Mat src, Mat dst) //fix contrast and fit brightness
{
	float contrast = 0.75;
	float sum;
	for(int i=0;i<src.rows;i++)
	{
		for(int j=0;j<src.cols;j++)
		{
			sum += dst.at<uchar>(i,j) - contrast*src.at<uchar>(i,j);
		}
	}
	float brightness = sum/(dst.cols*dst.rows);
	return make_pair(contrast,brightness);
}

Transform_block* generate_all_transform_blocks(Mat img, int src_size, int dst_size, int step)
{
	int factor = src_size/dst_size;
	int src_i = img.rows-src_size+1;
	int src_j = img.cols-src_size+1;
	/*int limit;
	if(src_i < src_j)
		limit = src_i;
	else
		limit = src_j;*/

	int size = 8*(1 + (int)(src_i/step))*(1 + (int)(src_j/step)); //no of transform blocks

	Transform_block* blocks = new Transform_block[size]; 
	int count = 0; //index for blocks array

	cout<<"src_i is "<<src_i<<" and src_j is"<<src_j<<"size is"<<size<<endl;

	Mat src,reSrc,S; //source block(domain), resized source block and Transformed source block
	for(int i=0;i<src_i;i+=step)
	{
		for(int j=0;j<src_j;j+=step)
		{
			//cout<<"hi";
			//img(Rect(j,i,src_size,src_size)).copyTo(src); //extract source block
			Rect roi(j,i,src_size,src_size);
			if(0<=roi.x && 0<=roi.width && roi.x+roi.width<=img.cols && 0<=roi.y && 0<=roi.height && roi.y+roi.height<=img.rows)
				src = img(roi);
			//src = extract(img,i,j,src_size,src_size);
			//resize(src,reSrc,Size(),factor,factor,INTER_CUBIC); //resizing domain block to range block size
			reSrc = reduce(src, factor);
			//cout<<"Cols and rows is"<<reSrc.cols<<reSrc.rows<<endl;
			//cout<<"i,j is"<<i<<","<<j<<endl;
			for (int k = 0; k < 4; k++)
			{
				float angle = k*90;
				//cout<<k;

				//only rotate
				S = rotate(reSrc,angle);
				Transform_params params(i,j,0,angle,0,0); //contrast & brightness 0 for now
				blocks[count].params = params; //noting data into blocks
				blocks[count].img = S;
				count++;

				//flip and rotate
				flip(reSrc,S,1); //flip(reflect) w.r.t Y-axis
				S = rotate(S,angle); //rotate
				//int some = S.at<uchar>(3,3);
				//cout<<some;
				params.direction = 1; //due to flip
				blocks[count].params = params; //noting data into blocks
				blocks[count].img = S;
				count++;
				//cout<<"count is "<<count;
			}

		}
	}
	//cout<<"out"<<endl;
	return blocks;
}

pair<Transform_params**,int*> compress(Mat img, int src_size, int dst_size, int step)
{
	int size = 8*(1 + (int)((img.cols-src_size+1)/step))*(1 + (int)((img.rows-src_size+1)/step)); //no of transform blocks
	Transform_block* blocks = new Transform_block[size]; 
	cout<<"size in compress is "<<size<<endl;
	int count = 0; //index for blocks array
	blocks = generate_all_transform_blocks(img,src_size,dst_size,step);
	cout<<"in"<<endl;
	int dst_i = img.rows - dst_size + 1; 
	int dst_j = img.cols - dst_size + 1;
	cout<<"dst_i is "<<dst_i<<" and dst_j is"<<dst_j<<endl;


	int width = (img.cols/dst_size);
	int height = (img.rows/dst_size);
	cout<<"width,height is"<<width<<","<<height<<endl; 
	Transform_params** params = new Transform_params*[height];
	for(int i=0;i<height;i++)
	{
		params[i] = new Transform_params[width];
	}

	Mat D; //destination block(range)
	float d,min_d;
	for(int i=0;i<dst_i;i+=dst_size)
	{
		for(int j=0;j<dst_j;j+=dst_size)
		{
			min_d = 100000.0; //initialising to a very large number
			//img(Rect(j,i,dst_size,dst_size)).copyTo(D); //extract destination block
			Rect roi(j,i,dst_size,dst_size);
			if(0<=roi.x && 0<=roi.width && roi.x+roi.width<=img.cols && 0<=roi.y && 0<=roi.height && roi.y+roi.height<=img.rows)
				D = img(roi);

			int i_ = i/dst_size;
			int j_ = j/dst_size;
			//cout<<"i,j is"<<i<<","<<j<<endl;
			for(count=0;count<size;count++)
			{
				Mat S = blocks[count].img;
				pair<float,float> con_bri = find_contrast_brightness(S,D);
				S = con_bri.first*S + con_bri.second;
				d = norm(D,S,NORM_L2);
				if(d < min_d)
				{
					min_d = d;
					params[i_][j_] = blocks[count].params; //copying better block's parameters
					params[i_][j_].contrast = con_bri.first;
					params[i_][j_].brightness = con_bri.second; 
				}
			}

		}
	}
	//cout<<sizeof(params[0][0])<<","<<sizeof(params);

	int length[2]; //to save size of params array
	length[0] = width;
	length[1] = height;
	return make_pair(params,length);
}

void decompress(pair<Transform_params**,int*> params, int src_size, int dst_size, int step, int iters=8)
{
	// size of params array
	//int cols = sizeof(params[0])/sizeof(params[0][0]);
	//int rows = (sizeof(params)/sizeof(params[0][0]))/cols;
	int cols = params.second[0];
	int rows = params.second[1];
	cout<<"rows,cols"<<rows<<","<<cols<<endl;
	//size of image
	int width = dst_size*cols;
	int height = dst_size*rows;
	Mat img = Mat::ones(height,width,CV_8UC1)*50;
	//cout<<"image rows,cols"<<img.rows<<","<<img.cols<<endl;
	Mat cur_img = Mat::zeros(height,width,CV_8UC1);
	Mat src,reSrc,dst;
	int factor = src_size/dst_size;

	for(int count=0;count<iters;count++)
	{
		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<cols;j++)
			{
				int m = params.first[i][j].k;
				int n = params.first[i][j].l;
				//cout<<"m,n"<<m<<","<<n;
				//cout<<" i,j"<<i<<","<<j;
				//img(Rect(n,m,src_size,src_size)).copyTo(src); //extract source block
				Rect roi(n,m,src_size,src_size);
				if(0<=roi.x && 0<=roi.width && roi.x+roi.width<=img.cols && 0<=roi.y && 0<=roi.height && roi.y+roi.height<=img.rows)
					src = img(roi);
				//resize(src,reSrc,Size(),factor,factor,INTER_CUBIC); //resizing domain block to range block size
				reSrc = reduce(src,factor);
				//cout<<"reduced rows,cols"<<reSrc.rows<<","<<reSrc.cols;
				if(params.first[i][j].direction == 1)
				{
					flip(reSrc,reSrc,1); //flip(reflect) w.r.t Y-axis
					//cout<<" flipped rows,cols"<<reSrc.rows<<","<<reSrc.cols<<endl;
					dst = rotate(reSrc,params.first[i][j].angle); //rotate
				}
				else
				{
					dst = rotate(reSrc,params.first[i][j].angle); //rotate
				}
				dst = dst*params.first[i][j].contrast + params.first[i][j].brightness;
				dst.copyTo(cur_img(Rect(j*dst_size,i*dst_size,dst_size,dst_size))); //copying transformed domain(i.e., ranage) to image
				img = cur_img;
			}
		}
	}
	namedWindow("image_final",CV_WINDOW_AUTOSIZE);
	imshow("image_final",cur_img);
	waitKey(0);

}

int main(int argc,char** argv)
{
	Mat img = imread("lena.jpg",0); //reading the source file

	if( img.empty() )// Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	namedWindow("image_initial",CV_WINDOW_AUTOSIZE);
	imshow("image_initial",img);
	waitKey(200);

	pair<Transform_params**,int*> parameters = compress(img,8,4,8);
	cout<<"Out of compress"<<endl;
	cout<<"width,height"<<parameters.second[0]<<","<<parameters.second[1];
	decompress(parameters,8,4,8,8);


	return 0;
}