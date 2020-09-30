//Q2:  WAP Write a game program....

#include<stdio.h>
#include<time.h>
void gamerules()
{
 printf("Welcome User ,\nGame rules: ");
 printf("\na) Enter Size of Square Matrix [n][n]\nb) Than insert data into matrix a/c to index");
 printf("\n\t a timer will start counting time(T) (in seconds)\n");
 printf("c) Matrix will be shown on screen, press 1 change, press 0 to continue\n");
 printf("d) You will be asked to enter no. of elememts(P) you wanna change ,if you press 1 ,\n");
 printf("e) Your Score will be calculated as follows : \n");
 printf("\tScore = 0 if it is not a magic square. Otherwise score = T + P*100,\n");
 printf("\twhere T is the time required in second and P is the number of alteration of entries\n");
 printf("\nThe player's performance will be judged by the minimum score achieved, other than zero.\n\n");
}
void main()
{   gamerules();
    int n,i,j,score,loop;
    int sum1=0,sum2=0,count=0,t=0,p=0;
    // a.) Read the size of the square matrix.
    printf("\nEnter the Size(n) of Square Matrix : ");
    scanf("%d",&n);
    int a[n][n],r[n],c[n];
    // b.) Display a square matrix (now it is blank) of N X N.
    printf("Here is yout  Blank Matrix with index a(raw,column) : \n");
    for(i=0;i<n;i++)
    {   
        printf("\n[\t");
        for(j=0;j<n;j++)
        {
            printf("a(%d,%d)\t\t",i+1,j+1);
        }
        printf("]");
    }
    printf("\n");
    // start counting time 
    clock_t start, end;
    start = clock();
    // c.) Allow the player to insert data into the matrix as displayed .
    for(i=0;i<n;i++)
    {   printf("Enter the elements of raw : %d \n",i+1);
        for(j=0;j<n;j++)
        {   
            printf("a(%d,%d) : ",i+1,j+1);
            scanf("%d",&a[i][j]);
        }
    }
    //d.)You should give a chance to confirm and to alter the previous entries, if desire)
    while(1)
    {   
        printf("Here is your matrix that you have Entered : \n");
        for(i=0;i<n;i++)
        {   
            printf("\n[\t");
            for(j=0;j<n;j++)
            {
                printf("%d\t\t",a[i][j]);
            }
            printf("]");
        }
        printf("\n Do you wanna make changes into matrix  ? \n Press [1 = YES] ,or [0 = NO] : ");
        scanf("%d",&loop);
        switch(loop)
        {
            case 1:
            {   int p1;
                // printf("\n You took %d seconds",t);  
                printf("\nRemember Clock is ticking, Complete Fast\nHow many Elememts do you wanna Change ? : ");
                scanf("%d",&p1);
                if(p1>0 && p1 <=n*n)
                {
                    for(int z=0;z<p1;z++)
                    {
                        printf("Enter Raw No. :");
                        scanf("%d",&i);
                        printf("Enter Column No. :");
                        scanf("%d",&j);
                        printf("Enter the Element a(%d,%d): ",i,j);
                        scanf("%d",&a[i-1][j-1]);
                    }
                    end = clock();
                    t = (end - start) / CLOCKS_PER_SEC; // updating time taken
                    p=p+p1;  // updating no. of alteration
                }
                else{
                    printf("Must be in the range [ 0< P <=n*n] \n");
                }
                continue; //inside case 1
            }
           /* case 0:
            {
                printf("\nNice Move\n");
                end = clock();
                t = (end - start) / CLOCKS_PER_SEC;
                break;
            }
           */
        }
        end = clock();
        t = (end - start) / CLOCKS_PER_SEC;
        break;
    }   // end of loop
    // magic Square calculation :)
    for(i=0;i<n;i++)
    {
        sum1=sum1+a[i][i]; // sum of diagonal 1
        sum2=sum2+a[i][n-1-i]; //sum of diagonal 2
    }
    if(sum1==sum2) // checking diagonal first
    {   
        // taking sum of each raw r[n] , and each column c[n]
        for(i=0;i<n;i++)
        {   
            int sum=0,jum=0; 
            for(j=0;j<n;j++)
            {   
                sum=sum+a[i][j]; 
                jum=jum+a[j][i];
            }
            r[i]=sum;   
            c[i]=jum; 
        }
        // equating sum of each raw and column
        for(i=0;i<n-1;i++)
        {   
            if(r[i]==r[i+1] && c[i]==c[i+1])
            {
                count=count+1; // no. of comparision 
            }
        }
        if(count==n-1)
        {
            printf("\nYour Matrix is a Magic Square :) \n");
            // e.) count the score as : score = t + p*100,
            // t= time taken (in seconds), p = no. of alteration.
            // here, I am taking alteration as : no of elememts changed. 
            score = t + p*100; // 
            printf("Your Score is : %d \n",score);
        }
        else
        {   // e.) Score = 0 if it is not a magic square.
            printf("It is not a Magic Square Matrixx \n Score = 0 \n");
        }
    }
    else
    {
        printf("!!!  It is not a Magic Square :_:  !!! \n Score = 0 \n ");
    }
}