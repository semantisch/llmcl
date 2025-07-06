% https://daydreampuzzles.com/logic-puzzles/diy-did-we-do-this/

%*
Deanne and Michael are in the middle of a DIY home renovation project that started off well but turned a little sour as they found they have very different approaches to practical tasks! To finish off the rest of the work (and save their relationship), they have just received quotes from various tradespeople. Can you match the correct contact person with their trade, and the amount they quoted?
*%

input(name,1,trent).
input(name,2,arthur).
input(name,3,andreas).
input(name,4,casey).

input(trade,1,plumber).
input(trade,2,carpenter).
input(trade,3,floor_sander).
input(trade,4,electrician).

input(quote,1,2500).
input(quote,2,4000).
input(quote,3,5000).
input(quote,4,6500).

index(name).

% 1) Arthur was the most highly recommended floor sander in the area, which led to further strain on Deanne and Michael's relationship once they found out he is fully booked for 6 months. Deanne had wanted to contact Arthur sooner, but Michael was still deliberating over flooring options.

clue(1,same).
object(1,1,name,arthur).
object(1,2,trade,floor_sander).

% 2) Of the Carpenter and the Electrician, one is Andreas and the other quoted $4,000 exactly.

clue(2,same).
object(2,1,trade,(carpenter;electrician)).
object(2,2,name,andreas).

clue(3,same).
object(3,1,trade,(carpenter;electrician)).
object(3,2,quote,4000).

% 3) Trent's quote was neither the highest nor the lowest of the four.

clue(4,same).
object(4,1,name,trent).
object(4,2,quote,(4000;5000)).

% 4) Casey quoted more than the Electrician, but less than Andreas.

clue(5,less).
object(5,1,trade,electrician).
object(5,2,name,casey).
object(5,3,name,andreas).
target(5,quote).
