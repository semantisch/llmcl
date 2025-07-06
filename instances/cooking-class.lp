% https://daydreampuzzles.com/logic-puzzles/cooking-class/

%*
Last night three friends who have just taken a cooking class held a dinner to practice their new skills. Can you identify which person brought which main course, and the dessert they each made for the evening?
*%

input(name,1,fiona).
input(name,2,tobias).
input(name,3,leanne).

input(dish,1,beef).
input(dish,2,gado).
input(dish,3,shrimps).

input(dessert,1,creme).
input(dessert,2,tiramisu).
input(dessert,3,cake).

index(name).

% 1) The three people are Tobias, the one who brought the Gado Gado, and the one who brought the Raspberry Tiramisu.

clue(1,diff).
object(1,1,name,tobias).
object(1,2,dish,gado).
object(1,3,dessert,tiramisu).

% 2) Fiona did not bring the Beef Wellington.

clue(2,diff).
object(2,1,name,fiona).
object(2,2,dish,beef).

% 3) The Shrimp Ceviche was brought by either Tobias or the person who brought the Creme Brulee.

clue(3,same).
object(3,1,dish,shrimps).
object(3,2,name,tobias).
object(3,2,dessert,creme).

clue(4,diff).
object(4,1,name,tobias).
object(4,2,dessert,creme).
