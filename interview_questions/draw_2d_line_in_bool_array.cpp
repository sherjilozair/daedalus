//  Draw a line on 2D array of boolean. You will be given start point and end point co-ordinates.

// this works fine for m < 1. for m > 1, mirror this algorithm with y <-> x

#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

void draw(bool arr[50][50], int x0, int y0, int x1, int y1, int sizex, int sizey){
	double m = ((double) y1 - y0) / (x1 - x0);
	double y = y0;
	for(int x = x0 + 1; x <= x1; x++){
		arr[(int) round(y)][x] = true;
		y += m;
	}
}

int main(){
	bool arr[50][50];
	for(int y = 0; y < 50; y++){
		for(int x = 0; x < 50; x++){
			arr[y][x] = false;
		}
		cout << endl;
	}
	draw(arr, 5, 45, 40, 5, 50, 50);
	for(int y = 0; y < 50; y++){
		for(int x = 0; x < 50; x++){
			if(arr[y][x])
				cout << "#";
			else
				cout << " ";
		}
		cout << endl;
	}

	return 0;
}