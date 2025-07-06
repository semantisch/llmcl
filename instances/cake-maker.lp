% https://daydreampuzzles.com/logic-puzzles/cake-maker/

%*
Lucy is a sought-after cake maker in the friendly small town of Plumpton. There aren't enough residents in the town to run her business full-time, so she bakes and decorates the cakes in the evenings after her main job. Tomorrow night, Lucy will be busy with three cakes to prepare - all different flavors and occasions! From the clues given, can you determine which client (identified by first name) ordered which type of cake, what the occasion is, and the price each customer will pay?
*%

input(name,1,sally).
input(name,2,michael).
input(name,3,emma).

input(flavor,1,chocolate).
input(flavor,2,carrot).
input(flavor,3,vanilla).

input(occasion,1,graduation).
input(occasion,2,birthday).
input(occasion,3,farewell).

input(price,1,35).
input(price,2,37).
input(price,3,39).

index(name).

% 1) Emma ordered carrot cake for an event she is hosting at her home, but it did not cost $37.00.

clue(1,same).
object(1,1,name,emma).
object(1,2,flavor,carrot).

clue(2,diff).
object(2,1,name,emma).
object(2,2,price,37).

% 2) Attendees of the farewell party will each be enjoying a big slice of vanilla cake.

clue(3,same).
object(3,1,occasion,farewell).
object(3,2,flavor,vanilla).

% 3) The cake Michael ordered cost $2.00 less than the cake that is for a birthday.

clue(4,(less;next)).
object(4,1,name,michael).
object(4,2,occasion,birthday).
target(4,price).

% 4) The chocolate cake is Lucy's specialty, which she currently has discounted to $35.00.

clue(5,same).
object(5,1,flavor,chocolate).
object(5,2,price,35).
