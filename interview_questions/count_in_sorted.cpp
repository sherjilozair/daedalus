// Find count of a number in sorted array.
// Use BS to find leftmost num and rightmost num.
// subtract and add one, to get answer.

#include <iostream>
#include <vector>
using namespace std;


int bsl(int arr[], int num, int start, int end, int size){
	int mid = start + (end - start)/2;
	if (arr[mid] == num && (mid == 0 || arr[mid-1] != num))
		return mid;
	else if (arr[mid] < num)
		return bsl(arr, num, mid+1, end, size);
	else if (arr[mid] >= num)
		return bsl(arr, num, start, mid, size);
	return -1;
}

int bsr(int arr[], int num, int start, int end, int size){
	int mid = start + (end - start)/2;
	if (arr[mid] == num && (mid == size || arr[mid+1] != num))
		return mid;
	else if (arr[mid] <= num)
		return bsr(arr, num, mid+1, end, size);
	else if (arr[mid] > num)
		return bsr(arr, num, start, mid, size);
	return -1;
}

int count(int arr[], int num, int size){
	int l = bsl(arr, num, 0, size, size);
	int r = bsr(arr, num, 0, size, size);
	return r - l + 1;
	
}
int main(){
	int a = 0;
	int arr[] = {1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5};
	int ans = count(arr, 3, 19);
	cout << ans << endl;
	return 0;
}