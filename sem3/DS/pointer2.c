/* p2.c
Find out (add code to print out) the address of the variable x in foo1, and the
variable y in foo2. What do you notice? Can you explain this?
*/
#include <stdio.h>
void foo1(int xval)
{
    int x;
    x = xval;
    printf("0x%p is the address of x = %d\n",&x,x);
    //int *p2;
    //p2 = &x;
    //printf("0x%p (withe the help of *p2)is the address of x = %d\n",p2,x);
    // Doubt : when I use pointer address changes ? 
    /* print the address and value of x here */
}
void foo2(int dummy)
{   
    int y;
    // Doubt : why the address is same ?
    // and value also if I dont ..
    // y = dummy;
    printf("0x%p is the address of y = %d\n",&y,y);
    //int *p1;
    //p1 = &y;
    //printf("0x%p (withe the help of *p1) is the address of x = %d\n",p1,y);
    /* print the address and value of y here */
}
int main()
{   
    printf("\n");
    foo1(7);
    foo2(11);
    return 0;
}