/* p3.c
The program below uses pointer arithmetic to determine the size of a 'char'
variable. By using pointer arithmetic we can find out the value of 'cp' and
the value of 'cp+1'. Since cp is a pointer, this addition involves pointer
arithmetic: adding one to a pointer makes the pointer point to the next
element of the same type.
For a pointer to a char, adding 1 really just means adding 1 to the address,
but this is only because each char is 1 byte.
1. Compile and run the program and see what it does.
2. Write some code that does pointer arithmetic with a pointer to an
int and determine how big an int is.
3. Same idea – figure out how big a double is, by using pointer
arithmetic and printing out the value of the pointer before and
after adding 1.
4. What should happen if you added 2 to the pointers from exercises
1 through 3, instead of 1? Use your program to verify your
answer.
*/

#include <stdio.h>
int main( )
{   
    printf("\n1.) trying Char \n"); 
    char c = 'Z';
    char *cp = &c; 
    printf("The character at cp is %c\n", *cp);
    printf("cp is 0x%08x\n", cp);
      //  printf("cp is %d\n", cp); // wit %d //doubt : what is this int 
    printf("Pointer arithmetic - see what cp+1 is\n");
    cp = cp+1;
    printf("cp is 0x%08x\n", cp); // its increasing in hexa, like b k baad c
    //printf("cp is %d\n", cp); // its increasing like norma
    /* Do not print *cp, because it points to
    memory space not allocated to your program */
    printf("\n2.) trying int \n"); 
    int a = 10;
    int *p1 = &a;
    printf("The integer at cp is %d\n", *p1);
    printf("p1 is 0x%08x\n", p1);
     //   printf("p1 is %d\n", p1);
    printf("Pointer arithmetic - see what p1 + 1 is\n");
    p1 = p1 +1;
    printf("p1 is 0x%08x\n", p1); // its increasing in hexa, like b k baad c
    // printf("p1 is %d\n", p1); // its increasing like norma
    //conclusion +4 ka increament on hex and deci (%d) both
    printf("\n3.) trying double \n"); 
    double b = 10.7854;
    double *pd = &b;
    printf("The double at pd is %.4f\n", *pd);
    printf("pd is 0x%08x\n", pd);
        printf("pd is %d\n", pd);
    printf("Pointer arithmetic - see what pd + 1 is\n");
    pd = pd +1;
    printf("pd is 0x%08x\n", pd); // its increasing in hexa, like b k baad c
        printf("pd is %d\n", pd); // its increasing like norma
    //conclusion +8 ka increament on hex and deci (%d) both
// 4.
    printf("\n4.) trying int with + 2 gap\n"); 
    int a2= 10;
    int *p2 = &a2;
    printf("The integer at a2 is %d\n", *p2);
    printf("p2 is 0x%08x\n", p2);
    
    printf("Pointer arithmetic - see what p2 + 1 is\n");
    p2 = p2 +1;
    printf("p2 is 0x%08x\n", p2); 
    //
    printf("\n5.) trying double + 2 gap\n"); 
    double b2 = 12.7854;
    double *pd2 = &b2;
    printf("The double at pd2 is %.4f\n", *pd2);
    printf("pd2 is 0x%08x\n", pd2);
        printf("pd2 is %d\n", pd2);
    //
    printf("Pointer arithmetic - see what pd2 + 2 is\n");
    pd2 = pd2 + 2;
    printf("pd2 is 0x%08x\n", pd2);
        printf("pd2 is %d\n", pd2);
    //gives gap of 16 :)

    
    return 0;
}