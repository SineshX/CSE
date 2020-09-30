//Q1: 
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

void main()
{   
    float score[30][6];
    int lower=24,upper=100;       //can we take float :_:
    srand(time(0));             
    for(int i=0;i<30;i++) 
    {   
        for(int j=0;j<6;j++)
        {
             int num = (rand() % (upper-lower +1)) + lower;
             score[i][j]=num;
        }
    }
    //Average Function
    float average(int test)
    {
        float sum =0,avg=0;
        for(int i=0;i<30;i++)  
        { 
            sum = sum + score[i][test]; //for test 1 [1-30][0]
        }
        avg = sum/30; 
        printf("\n   Average grade of test %d = %.2f ",test+1,avg); 
        return avg;
    }

    // a: average grade of each test
    printf("\nA:");
    for(int k=0;k<6;k++)
    {
        average(k);
    }
    
    // b: Average grade of highest of five out of six subjcts 
    float final_grade(int s_number) //s_number = student number
    {   float favg=0,sum=0,min=0;
        min=score[s_number][0];
        for(int i=0;i<6;i++)
        {   
            if (score[s_number][i]<min)
            {
                min = score[s_number][i];  //calculating minimum marks out of 6
            }
            sum=sum+score[s_number][i];
        }
        favg = (sum/5)-(min/5);
        return favg;
    }

    //b: final grade of each student
    printf("\nB: ");
    for(int i=0;i<30;i++)
    {
    printf("\n  Final average grade of student %d = %.2f ",i+1,final_grade(i));
    }

    //c: number of failed students (i.e. 60 se kam marks wale)
    int count = 0;
    for(int i=0;i<30;i++)
    {
        if(final_grade(i)<60)
        {
            count = count + 1;
        }
    }
    printf("\nC:");
    printf("\n  Number of Students failed = %d ",count);

    // d: Average of final grade 
    float avgfgrade=0;
    for(int i=0;i<30;i++)  
    {
        avgfgrade = avgfgrade + final_grade(i);
    }
    printf("\nD:");
    printf("\n  Average of final grades = %.2f \n\n",avgfgrade/30); 
}