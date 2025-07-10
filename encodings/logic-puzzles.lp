%* DATA *%

% Please feel free to use, modify or drop the below auxiliary predicates...

value(A,V) :- input(A,N,V).
number(N)  :- input(A,N,V).

%* IMPLEMENTATION *%

{assign(A,V,N):number(N)} = 1 :- value(A,V).
assigned(ID,POS,N) :- object(ID,POS,A,V), assign(A,V,N).

:- assign(A,V1,N),assign(A,V2,N),V1!=V2.

% DIFF CONSTRAINT
:- assigned(ID,POS1,N1), assigned(ID,POS2,N2), clue(ID,diff), N1=N2,POS1!=POS2.

% SAME CONSTRAINT
% Naive encoding (not working)
%:- assigned(ID,POS1,N1), assigned(ID,POS2,N2), clue(ID,same), N1!=N2, POS1!=POS2.

% More involved (seems to work - but still false)
%aux(ID,POS1) :- assigned(ID,POS1,N1), #count{N2:assigned(ID,POS2,N2), POS1!=POS2, N1=N2} >= 1, clue(ID,same).
%:- assigned(ID,POS,_), not aux(ID,POS), clue(ID,same).

% Even more involved (from Martin)
aux_same(ID,N) :- clue(ID,same), number(N), object(ID,POS,_,_), not assigned(ID,POS,N).
:- clue(ID,same), aux_same(ID,N):number(N).

% Positive other side (seems also to work - works on all examples):
%check_same(ID) :- number(N), #count{POS:assigned(ID,POS,N)} = ALPHA, ALPHA = #count{POS:object(ID,POS,_,_)}, clue(ID,same).
%:- clue(ID,same), not check_same(ID).

% Less:
:- clue(ID,less), target(ID,A), assign(A,V1,N1), assign(A,V2,N2), input(A,ORD1,V1), input(A,ORD2,V2), assigned(ID,POS1,N1), assigned(ID,POS2,N2), POS1 < POS2, ORD1 >= ORD2.

% Next:
:- clue(ID,next), target(ID,A), assign(A,V1,N1), assign(A,V2,N2), input(A,ORD1,V1), input(A,ORD2,V2), assigned(ID,POS1,N1), assigned(ID,POS2,N2), |POS1 - POS2| = POSDIFF, |ORD1 - ORD2| != POSDIFF.

% MATCH:
match(A1,V1,A2,V2) :- assign(A1,V1,N), assign(A2,V2,N), A1 < A2.



%* DISPLAY *%

#show match/4.
