#include <avr/io.h>
#include <stdbool.h>
int main (void)
{

	 bool A=0,B=0,C=0,F=0;
	  DDRB =  0b00100000;  
	   DDRD =  0b11100011;
	    PORTD = 0b00011100;   

	    while(1)
	    {  
		    A = (PIND & (1 << PIND2)) == (1 << PIND2);
		    B = (PIND & (1 << PIND3)) == (1 << PIND3);                                  C = (PIND & (1 << PIND4)) == (1 << PIND4);
		    F = ((!C)|(A&B));
		    PORTB |= (F<< 5);
	    }
	    return 0;
}
