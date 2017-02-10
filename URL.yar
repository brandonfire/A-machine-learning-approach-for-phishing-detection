

rule pat 
{
    strings:
       
        $a = "@"

    condition:
        $a
}

rule slash 
{
    strings:
        $a = "//"
    condition:
        #a > 1


}

rule https 

{
    strings:
        $a = "https://"
    condition:
        $a at 0
}


rule phishinghttps 

{
    strings:
        $a = "https"
    condition:
        $a in (6..filesize)
}

