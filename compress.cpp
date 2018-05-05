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
					reduced_img.at<float>(i,j) += img.at<float>(k,l);
				}
			}

			reduced_img.at<float>(i,j) = reduced_img.at<float>(i,j)/(factor*factor);
		}
	}

	return reduced_img;
}

Mat rotate(Mat img,float angle)
{
	Point2f centre(img.cols/2.0F,img.rows/2.0F); //create point about which it rotates
	Mat rot_matrix = getRotationMatrix2D(centre, angle, 1.0); //Rotation matrix
	Mat rotated_img(Size(img.size().height, img.size().width), img.type());
	warpAffine(img, rotated_img, rot_matrix, img.size()); //transformation

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
			sum += dst.at<float>(i,j) - contrast*src.at<float>(i,j);
		}
	}
	float brightness = sum/(dst.cols*dst.rows);
	return make_pair(contrast,brightness);
}

Transform_block* generate_all_transform_blocks(Mat img, int src_size, int dst_size, int step)
{
	int factor = src_size/dst_size;
	int src_i = img.rows-src_size;
	int src_j = img.cols-src_size;

	int size = 8*(1 + (int)(src_i/step))*(1 + (int)(src_j/step)); //no of transform blocks

	Transform_block* blocks = new Transform_block[size]; 
	int count = 0; //index for blocks array

	Mat src,reSrc,S; //source block(domain), resized source block and Transformed source block
	for(int i=0;i<src_i;i+step)
	{
		for(int j=0;j<src_j;j+step)
		{
			img(Rect(i,j,src_size,src_size)).copyTo(src); //extract source block
			//resize(src,reSrc,Size(),factor,factor,INTER_CUBIC); //resizing domain block to range block size
			reSrc = reduce(src, factor);
			for (int k = 0; k < 4; k++)
			{
				float angle = k*90;

				//only rotate
				S = rotate(reSrc,angle);
				Transform_params params(i,j,0,angle,0,0); //contrast & brightness 0 for now
				blocks[count].params = params; //noting data into blocks
				blocks[count].img = S;
				count++;

				//flip and rotate
				flip(reSrc,S,1); //flip(reflect) w.r.t Y-axis
				S = rotate(S,angle); //rotate
				params.direction = 1; //due to flip
				blocks[count].params = params; //noting data into blocks
				blocks[count].img = S;
				count++;
			}

		}
	}

	return blocks;
}

Transform_params** compress(Mat img, int src_size, int dst_size, int step)
{
	int size = 8*(1 + (int)((img.cols-src_size)/step))*(1 + (int)((img.rows-src_size)/step)); //no of transform blocks
	Transform_block* blocks = new Transform_block[size]; 
	int count = 0; //index for blocks array
	blocks = generate_all_transform_blocks(img,src_size,dst_size,step);

	int dst_i = img.rows - dst_size; 
	int dst_j = img.cols - dst_size;

	const int width = (img.cols/dst_size);
	const int height = (img.rows/dst_size); 
	Transform_params** params = new Transform_params*[height];
	for(int i=0;i<height;i++)
	{
		params[i] = new Transform_params[width];
	}

	Mat D; //destination block(range)
	float d,min_d;
	for(int i=0;i<dst_i;i+dst_size)
	{
		for(int j=0;j<dst_j;j+dst_size)
		{
			min_d = 100000.0; //initialising to a very large number
			img(Rect(i,j,dst_size,dst_size)).copyTo(D); //extract destination block
			for(count=0;count<size;count++)
			{
				Mat S = blocks[count].img;
				pair<float,float> con_bri = find_contrast_brightness(S,D);
				S = con_bri.first*S + con_bri.second;
				d = norm(D,S,NORM_L2);
				if(d < min_d)
				{
					min_d = d;
					params[i][j] = blocks[count].params; //copying better block's parameters
					params[i][j].contrast = con_bri.first;
					params[i][j].brightness = con_bri.second; 
				}
			}

		}
	}

	return params;
}

void decompress(Transform_params** params, int src_size, int dst_size, int step, int iters=8)
{
	// size of params array
	int cols = sizeof(params[0])/sizeof(params[0][0]);
	int rows = (sizeof(params)/sizeof(params[0][0]))/cols;
	//size of image
	int width = dst_size*cols;
	int height = dst_size*rows;
	Mat img = Mat::ones(height,width,CV_8UC1)*150;
	Mat cur_img = Mat::zeros(height,width,CV_8UC1);
	Mat src,reSrc,dst;
	int factor = src_size/dst_size;

	for(int count=0;count<iters;count++)
	{
		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<cols;j++)
			{
				int m = params[i][j].k*step;
				int n = params[i][j].l*step;
				img(Rect(m,n,src_size,src_size)).copyTo(src); //extract source block
				//resize(src,reSrc,Size(),factor,factor,INTER_CUBIC); //resizing domain block to range block size
				reSrc = reduce(src,factor);
				if(params[i][j].direction == 1)
				{
					flip(reSrc,reSrc,1); //flip(reflect) w.r.t Y-axis
					dst = rotate(reSrc,params[i][j].angle); //rotate
				}
				else
				{
					dst = rotate(reSrc,params[i][j].angle); //rotate
				}
				dst = dst*params[i][j].contrast + params[i][j].brightness;
				dst.copyTo(cur_img(Rect(i,j,dst_size,dst_size))); //copying transformed domain(i.e., ranage) to image
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
	Mat img = imread("pup.jpg",0); //reading the source file

	if( img.empty() )// Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	namedWindow("image_initial",CV_WINDOW_AUTOSIZE);
	imshow("image_initial",img);
	waitKey(200);

	Transform_params** parameters = compress(img,8,4,8);
	decompress(parameters,8,4,8);


	return 0;
}