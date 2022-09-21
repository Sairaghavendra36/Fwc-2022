.include "/home/user/m328Pdef.inc"

ldi r16,0b11100011 ;identifying input pins 2,3,4
out DDRD,r16	    ;declaring pins as input
ldi r16,0b11111111 ;activating pullups for pins 2,3,4
out PORTD,r16

ldi r17,0b00100000 ;identifying output pin as 1
out DDRB,r17		;declaring pin as output
loopw:
in r16, PIND


ldi r17, 0b00001000 
and r17,r16 ;r17 -B
lsl r17
lsl r17
ldi r18, 0b00000100
and r18,r16;r18 -A
lsl r18
lsl r18
lsl r18
and r17,r18	
ldi r21, 0b00010000 
and r21,r16 ; r21-c 
ldi r20, 0b00010000
eor r21,r20
lsl r21
or r17,r21			;c'+ab
out PORTB ,r17 
rcall loopw
strt:
 rjmp strt

;comp:
;		mov r20,r21
;		ldi r21, 0b00010000
;		eor r21,r20
;		ret
