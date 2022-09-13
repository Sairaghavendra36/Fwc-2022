#include <Arduino.h>
int A,B,C;
int F;
void disp_comb(int F)
{
	digitalWrite(7,F);
}
void setup(){
	pinMode(7, OUTPUT);
	pinMode(2, INPUT);
	pinMode(3, INPUT);
	pinMode(4, INPUT);
}
void loop(){
	A = digitalRead(2);
	B = digitalRead(3);
	C = digitalRead(4);
	F = (!C) || (A&&B);
	disp_comb(F);
}
