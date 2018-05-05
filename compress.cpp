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

Mat rotate(Mat img,float angle)
{
	Point2f centre(img.cols/2.0F,img.rows/2.0F); //create point about which it rotates
	Mat rot_matrix = getRotationMatrix2D(centre, angle, 1.0); //Rotation matrix
	Mat rotated_img(Size(img.size().height, img.size().width), img.type());
	warpAffine(img, rotated_img, rot_matrix, img.size()); //transformation

	return rotated_img;
}

pair<float,float> find_contrast_brightness(Mat src, Mat dst)
{

}

Transform_block* generate_all_transform_blocks(Mat img, int src_size, int dst_size, int step)
{
	float factor = dst_size/src_size;
	int src_i = img.cols-src_size;
	int src_j = img.rows-src_size;

	int size = 8*(1 + (int)(src_i/step))*(1 + (int)(src_j/step)); //no of transform blocks

	Transform_block* blocks = new Transform_block[size]; 
	int count = 0; //index for blocks array

	Mat src,reSrc,S; //source block(domain), resized source block and Transformed source block
	for(int i=0;i<src_i;i+step)
	{
		for(int j=0;j<src_j;j+step)
		{
			img(Rect(i,j,src_size,src_size)).copyTo(src); //extract source block
			resize(src,reSrc,Size(),factor,factor,INTER_CUBIC); //resizing domain block to range block size
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

Transform_params* compress(Mat img, int src_size, int dst_size, int step)
{
	int size = 8*(1 + (int)((img.cols-src_size)/step))*(1 + (int)((img.rows-src_size)/step)); //no of transform blocks
	Transform_block* blocks = new Transform_block[size]; 
	int count = 0; //index for blocks array
	blocks = generate_all_transform_blocks(img,src_size,dst_size,step);

	int dst_i = img.cols - dst_size; 
	int dst_j = img.rows - dst_size;

	int number = (img.cols/dst_size)*(img.rows/dst_size); //no of dst blocks
	Transform_params* params = new Transform_params[number];

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
					params[i*dst_size+j] = blocks[count].params; //copying better block's parameters
					params[i*dst_size+j].contrast = con_bri.first;
					params[i*dst_size+j].brightness = con_bri.second; 
				}
			}

		}
	}

	return params;
}

int main(int argc,char** argv)
{
	Mat img = imread("pup.jpg",0); //reading the source file

	if( img.empty() )// Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	namedWindow("image_initial",CV_WINDOW_NORMAL);
	imshow("image_initial",img);
	waitKey(0);


}