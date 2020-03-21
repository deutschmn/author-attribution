SELECT * FROM Articles WHERE ID_Article = 1860;

-- Article Categories
SELECT p.ID_Article, a.Category, COUNT(*) 'nr'
FROM Posts p INNER JOIN Annotations_consolidated a ON (p.ID_Post = a.ID_Post)
WHERE a.Value = 1 -- AND a.Category = 'SentimentNegative'
GROUP BY p.ID_Article, a.Category
HAVING COUNT(*) > 5
ORDER BY p.ID_Article, nr DESC;

-- User Categories
SELECT ID_User, a.Category, COUNT(*) 'nr'
FROM Posts p INNER JOIN Annotations_consolidated a ON (p.ID_Post = a.ID_Post)
WHERE a.Value = 1 AND (	a.Category = 'SentimentNegative' OR 
											a.Category = 'SentimentNeutral' OR
											a.Category = 'SentimentPositive')
--			AND p.ID_User = 3788
GROUP BY p.ID_User, a.Category
 HAVING COUNT(*) > 10
ORDER BY ID_User, nr DESC;