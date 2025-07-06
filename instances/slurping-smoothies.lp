% https://daydreampuzzles.com/logic-puzzles/slurping-smoothies/

%*
Anya has made many attempts at beginning healthy habits this year, each falling by the wayside after a few weeks at most! She has just renewed her determination to be healthier by starting the practice of drinking daily smoothies each morning. Each smoothie has a green vegetable, a type of nut, and a type of fruit, among other ingredients. She began on Monday and is four days in. Can you determine which ingredients were combined together, and the day Anya drank that smoothie?
*%

input(day,1,monday).
input(day,2,tuesday).
input(day,3,wednesday).
input(day,4,thursday).

input(fruit,1,mango).
input(fruit,2,banana).
input(fruit,3,apple).
input(fruit,4,pear).

input(vegetable,1,kale).
input(vegetable,2,spinach).
input(vegetable,3,collard).
input(vegetable,4,lettuce).

input(nut,1,pecans).
input(nut,2,almonds).
input(nut,3,macademias).
input(nut,4,cashews).

index(day).

% 1) The pecan smoothie did not contain mangoes nor bananas.

clue(1,diff).
object(1,1,nut,pecans).
object(1,2,fruit,(mango;banana)).

% 2) For the first two days of the week Anya had spinach and lettuce in her smoothies - but not necessarily in that order.

clue(2,same).
object(2,1,day,(monday;tuesday)).
object(2,2,vegetable,spinach).

clue(3,same).
object(3,1,day,(monday;tuesday)).
object(3,2,vegetable,lettuce).

% 3) The kale smoothie did not have macadamias in it, nor did the spinach smoothie.

clue(4,diff).
object(4,1,vegetable,(kale;spinach)).
object(4,2,nut,macademias).

% 4) On Thursday Anya drank a smoothie containing almonds.

clue(5,same).
object(5,1,day,thursday).
object(5,2,nut,almonds).

% 5) The banana smoothie was one day after the smoothie containing pecans, and two days after the apple smoothie.

clue(6,(less;next)).
object(6,1,fruit,apple).
object(6,2,nut,pecans).
object(6,3,fruit,banana).
target(6,day).

% 6) Anya enjoyed the combination of pear with lettuce more than she expected to.

clue(7,same).
object(7,1,fruit,pear).
object(7,2,vegetable,lettuce).
