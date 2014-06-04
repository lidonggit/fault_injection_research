#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( int argc, char *argv[] )
{
    if( argc != 3 )
    {
        printf("comp [file 1] [file 2] \n");
    }
    else
    {
 //       printf("\nStart to compare %s and %s ...\n", argv[1], argv[2]);
    }

    FILE *fp1,*fp2;
    int num1,num2;
    int counter = 0;
    int total_num = 0;
    int flag = 0;

    fp1 = fopen(argv[1], "rb");
    fp2 = fopen(argv[2], "rb");
    if( !fp1 || !fp2 )
    {
        printf("Seg Fault\n");
        return 1;
    }

    while ( fscanf(fp1, "%d", &num1)>0 && fscanf(fp2, "%d", &num2)>0 )
    { 
        total_num ++;
        
        if (num1 == num2)
        {
            flag = 1;
        }
        else
        {
            counter ++;
//          printf("%d %d\n", num1,num2);
            flag = 0;
        }
    }
    printf("%d %0.8f\n", counter, counter/(double)total_num);
}
