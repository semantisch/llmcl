% https://daydreampuzzles.com/logic-puzzles/egg-citing-easter-hunt/

%*
This Easter the Gould family are hosting a family gathering including an Easter Egg Hunt for the youngest children. The child who found the most eggs came first and won an extra special prize! From the clues given, can you figure out which position each child placed (based on the number of eggs they found), and the pattern and hiding place of each child's very first egg-citing find of the day?
*%

input(position,1,1).
input(position,2,2).
input(position,3,3).
input(position,4,4).

input(name,1,mandy).
input(name,2,thomas).
input(name,3,riley).
input(name,4,sunita).

input(pattern,1,dots).
input(pattern,2,swirls).
input(pattern,3,stars).
input(pattern,4,lines).

input(place,1,garden).
input(place,2,sandpit).
input(place,3,mailbox).
input(place,4,slide).

index(position).

% 1) The grandparents of the family, who hosted the event, have a mini playground in the background for the kids. The egg with the purple swirls was found under the slide.

clue(1,same).
object(1,1,pattern,swirls).
object(1,2,place,slide).

% 2) The child who placed 1st did not find the egg with the purple swirls.

clue(2,diff).
object(2,1,position,1).
object(2,2,pattern,swirls).

% 3) The egg that was cleverly placed in the mailbox was found quicker than expected! It was decorated with multi-colored dots.

clue(3,same).
object(3,1,place,mailbox).
object(3,2,pattern,dots).

% 4) Riley didn't find the most eggs, nor did Sunita.

clue(4,diff).
object(4,1,position,1).
object(4,2,name,(riley;sunita)).

% 5) The child who placed third found the egg with the blue and yellow stars pattern.

clue(5,same).
object(5,1,position,3).
object(5,2,pattern,stars).

% 6) Thomas, who did not place 4th, was either the child who found their first egg under the slide, or the child who found the egg with the green wavy lines.

clue(6,diff).
object(6,1,name,thomas).
object(6,2,position,4).

clue(7,same).
object(7,1,name,thomas).
object(7,2,place,slide).
object(7,2,pattern,lines).

clue(8,diff).
object(8,1,place,slide).
object(8,2,pattern,lines).

% 7) Sunita was either the child who placed 4th, or the one who found their first egg in the mailbox.

clue(9,same).
object(9,1,name,sunita).
object(9,2,position,4).
object(9,2,place,mailbox).

clue(10,diff).
object(10,1,position,4).
object(10,2,place,mailbox).

% 8) Riley found his first egg in the sandpit.

clue(11,same).
object(11,1,name,riley).
object(11,2,place,sandpit).

% 9) Thomas (who did not find his first egg in the herb garden) found more eggs than Riley, but less eggs than Mandy.

clue(12,diff).
object(12,1,name,thomas).
object(12,2,place,garden).

clue(13,less).
object(13,1,name,mandy).
object(13,2,name,thomas).
object(13,3,name,riley).
target(13,position).

% 10) Mandy did not find her first egg in the herb garden.

clue(14,diff).
object(14,1,name,mandy).
object(14,2,place,garden).
