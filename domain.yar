rule dash 
{
    strings:
       
        $a = "-"

    condition:
        $a
}

rule phishingsubdomain
{
    strings:
        $a = "."
    condition:
        #a >3
}

rule suspectsubdomain
{
    strings:
        $a = "."
    condition:
        #a ==3
}
