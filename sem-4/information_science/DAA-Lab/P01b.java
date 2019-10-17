/*

b. Write a Java program to implement the Stack using arrays.
Write Push(), Pop(), and Display() methods to demonstrate its working.

*/
import java.util.Scanner;
class Stack
{
	int top;
	int max;
	int[] stk; 
	Stack()
	{
		top=-1;
		max=5;
		stk  = new int[max];
	}
	public  void push()
	{
	int elem;	
	if(top==max-1)
		System.out.println("Stack overflow");
	else
	{ 
		System.out.println("Enter the element");
		Scanner in = new Scanner(System.in);
		elem = in.nextInt();
		top = top+1;
		stk[top]=elem;
		}
	}
	public void pop()
	{
		int a;
		if(top==-1)
			System.out.println("Stack Underflow");
		else
		{
			a=stk[top];
			top=top-1;
			System.out.println("Popped element is "+a);
		}		
	}
	public void display()
	{
		if(top==-1)
			System.out.println("Stack is empty");
		else
		{
			System.out.println("Stack elements are");
			for(int i=top; i>=0; i--)
				System.out.println(stk[i]);
		}
	}
}
public class P02
{
	public static void main(String[] args)
	{
		Scanner in = new Scanner(System.in);
		Stack obj=new Stack();
		while(true)
		{
			System.out.println("-----------Menu------------");
			System.out.println("1.Push\t2.Pop\t3.Display");
			System.out.println("---------------------------");
			System.out.println("Enter your choice");
			int ch = in.nextInt();
			switch(ch)
			{
				case 1:obj.push();
					break;
				case 2:obj.pop();
					break;
				case 3:obj.display();
					break;
				default:System.out.println("Invalid choice");
					return;
			
			}			
		}
    }
}
