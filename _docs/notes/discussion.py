
'''
      Why don't we adopt a general SQL function that wraps around insert, update, ect.
'''
# <JON>
#     this would look nicer:
#     ibs.db.insert(table='names', columns=('names_uid', 'names_text'), values=[0, '____'])
#
#     I think it might be better to specify insert as a function, which
#     takes some table (images) as an arg along with a tuple of of columns
#     then have the database control build the string.
#     This would allow for cleaner more reusable code
#     The error would also be generated on the fly and be much more
#     descriptive as well as not polluting the IBEISControl
# </JON>
#
# <JASON>
#     While this does indeed look nicer, it has issues with extensibility.
#     For example,
#
#         UPDATE chips
#             SET
#             chips.name_uid=
#             (
#                 SELECT names.name_uid
#                 FROM names
#                 WHERE name_text=?
#                 ORDER BY name_uid
#                 LIMIT 1
#             ),
#             WHERE chip_uid=?
#
#     We would be creating a lot of different ways to call the different commands.
#     The part that is keeping me from adopting such a structure is that the
#     commands would be super complex for some queries and we also have a very large
#     number of kinds of querries, each with their own structure.  Generalization becomes
#     a problem.  I have done the easier querries first, but there are a lot of queries
#     that will use a LEFT JOIN operation in the getters for efficiency's sake.  So if we
#     are going to have to keep a general function to handle these super complex queries,
#     why not make them all this form?  This way, all SQL is in the controller as opposed to
#     being dispursed between multiple files.  I'm not terribly happy with the non-general
#     nature, but having some of the queries in my head, it is going to be a huge pain to
#     generalize (not bringing into the fact that there are security / consistency issues).
#     I'm not terribly sold on not generalizing, but I'd like to re-evaluate after we have
#
#     A generic function would be great, but we would lose a lot of
#     extensibility in the process.  Yes, these commands are long and complex
#     but a "security" issue is also raised by passing to a generator function.
#     Furthermore, there is nothing to prevent us from doing complex error handling
#     in the commit function as it is.  The SQL object will throw errors if, for example,
#     the columns don't exist.
# </JASON>


        # Jon: I think we should be using the fetchmany command here
        # White iteration is efficient, I believe it still interupts
        # the sql work. If we let sql work uninterupted by python it
        # should go faster

        # Jason: That's fine, it will just be a bigger memory footprint
        # Speed vs Footprint

